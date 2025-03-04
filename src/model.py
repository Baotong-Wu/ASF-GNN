
import parse
from torch_geometric.nn import GCNConv
import collections
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

from utils import enPrint
from transformers import BartForConditionalGeneration
from dataProcessor import GraphDataset, data_preprocess, process_data
from embedConstruction import  predict_test
from parse import argParser
import pandas as pd
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
args = argParser()
DATASET = args.dataset
CUDA_AVAILABLE = torch.cuda.is_available()

base_path = data_preprocess(DATASET)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2 = BartForConditionalGeneration.from_pretrained('./outputs').to(device)

# Load data
review_data = pd.read_csv(base_path + f'{DATASET.lower()}_review_data.csv')

# embedding construction
user_preferences, item_ratings = predict_test(model2, device, review_data)
user_item_ratings = pd.read_csv(base_path + f'user_item_ratings.csv')
process_data(user_preferences, item_ratings, user_item_ratings)

user_preferences_tensor, item_ratings_tensor, \
    user_ids, item_ids, user_id_to_index, \
    item_id_to_index, rating_matrix, ideal_item_embeddings_tensor,\
    edge_index = process_data(user_preferences, item_ratings, user_item_ratings)

embedding_dim = item_ratings_tensor.shape[1]
print(f"Ideal item embeddings tensor shape: {ideal_item_embeddings_tensor.shape}")

torch.save(ideal_item_embeddings_tensor, 'ideal_item_embeddings_tensor.pt')
loaded_tensor = torch.load('ideal_item_embeddings_tensor.pt')

print(torch.equal(ideal_item_embeddings_tensor, loaded_tensor))  # Should output True

class ASFGNN(nn.Module):
    """
    Graph Neural Network Model - PDA-GNN
    """

    def __init__(self, graph: GraphDataset):
        super(ASFGNN, self).__init__()
        args = argParser()
        self.graph = graph

        # Parameter Loaded
        self.embeddingDim = 6
        self.hiddenDims = args.att_hidden_dims
        self.keepProb = args.keep_prob
        self.device = parse.DEVICE
        self.finalIntegration = args.final_integration
        self.attNorm = args.att_norm
        # numLayers stands for GCN propagation depth, identifying with the number of attributes
        self.numLayers = args.num_layers
        self.ifRegBehav = args.ifRegBehav
        self.ifRegEmbedding = args.ifRegEmbedding
        self.ifDropOut = args.ifDropOut

        self.gcn_layers = nn.ModuleList()
        for _ in range(self.numLayers):
            self.gcn_layers.append(GCNConv(self.embeddingDim, self.embeddingDim))
        self.userPrefEmbeddings = nn.Linear(6, self.embeddingDim).to(self.device)
        init.eye_(self.userPrefEmbeddings.weight)
        if self.userPrefEmbeddings.bias is not None:
            init.zeros_(self.userPrefEmbeddings.bias)
        self.itemRatingEmbeddings = nn.Linear(6, self.embeddingDim).to(self.device)
        init.eye_(self.itemRatingEmbeddings.weight)
        if self.itemRatingEmbeddings.bias is not None:
            init.zeros_(self.itemRatingEmbeddings.bias)

        self.activeLayer = nn.Sigmoid()

        self.attMlp = nn.Sequential(
            collections.OrderedDict([('attMLP-layer0',
                                      torch.nn.Linear(in_features=self.embeddingDim, out_features=self.hiddenDims[0]))])
        )
        for layerIdx in range(len(self.hiddenDims) - 1):
            linear = torch.nn.Linear(in_features=self.hiddenDims[layerIdx], out_features=self.hiddenDims[layerIdx + 1])
            self.attMlp.add_module('attMLP-layer{0}'.format(layerIdx + 1), linear)
        self.softMaxLayer = nn.Softmax(dim=1).to(self.device)

        dimInput, dimOutput = self.embeddingDim, self.embeddingDim
        self.attW = nn.Parameter(torch.zeros(size=(dimInput, dimOutput)))
        self.attA = nn.Parameter(torch.zeros(size=(2 * dimOutput, 1)))
        nn.init.xavier_uniform_(self.attW.data, gain=1.414)
        nn.init.xavier_uniform_(self.attA, gain=1.414)
        self.attW.to(device=self.device)
        self.attA.to(device=self.device)
        self.leakyReLU = nn.LeakyReLU(negative_slope=2e-1)

        enPrint("Model Loaded...")
        enPrint("Embedding Initialization Loaded...")

    def attLayer(self, inputTensor: torch.Tensor):
        """
        :param inputTensor  The input final tensor[B, n_b, D]
        :return             attention weighted tensor matrix
        """
        # batchSize: B
        batchSize = inputTensor.size()[0]
        # numBehavs: n_b
        numBehavs = inputTensor.size()[1]
        # dimInput: D, dimOutput: D'
        dimInput = inputTensor.size()[2]
        dimOutput = self.attW.size()[1]

        # [B, n_b, D'] -> [B, n_b, D']
        h = torch.matmul(inputTensor, self.attW)
        # [B, n_b * n_b, 2 * D'] -> [B, n_b, n_b, 2 * D']
        aInput = torch.cat([h.repeat(1, 1, numBehavs).view(batchSize, numBehavs * numBehavs, -1),
                            h.repeat(1, numBehavs, 1)], dim=2).view(batchSize, numBehavs, -1, 2 * dimOutput)
        # [B, n_b, n_b, 1] -> [B, n_b, n_b]
        attMat = self.leakyReLU(torch.matmul(aInput, self.attA)).squeeze(3)
        attMat = F.softmax(attMat, dim=2)
        h_prime = torch.matmul(attMat, h)

        return h_prime

    def propagation(self):
        """
        Multi-attribute GCN propagation
        :return: [ userFinalEmbedding, itemFinalEmbedding ]
        """
        finalEmbeds = []
        for behaviorDepth in range(self.numLayers + 1):
            # print("behaviorDepth:", behaviorDepth)
            singleBehaivorEmbed = self.singleBehaviorPropagation(behaviorDepth + 1, edge_index)
            finalEmbeds.append(singleBehaivorEmbed)
        finalEmbedding = None
        if self.finalIntegration == 'MEAN':
            finalEmbedding = torch.stack(finalEmbeds, dim=0)
            finalEmbedding = torch.mean(finalEmbedding, dim=0)
        elif self.finalIntegration == 'NONE':
            finalEmbedding = finalEmbeds[-1]
        elif self.finalIntegration == 'ATT':
            finalEmbedding = torch.stack(finalEmbeds, dim=1)
            if self.attNorm == 'GAT-like':
                finalEmbedding = self.attLayer(finalEmbedding)
                finalEmbedding = torch.sum(finalEmbedding, dim=1)
            else:
                attWeight = self.attMlp(finalEmbedding)
                if self.attNorm == 'SOFTMAX':
                    attWeight = self.softMaxLayer(attWeight)
                elif self.attNorm == 'SUM-RATIO':
                    attSum = torch.sum(attWeight, dim=1, keepdim=True)
                    attWeight = attWeight / attSum
                finalEmbedding = finalEmbedding.permute(0, 2, 1)
                finalEmbedding = torch.matmul(finalEmbedding, attWeight).squeeze()
        userFinalEmbedding, itemFinalEmbedding = torch.split(finalEmbedding, [self.graph.numUsers, self.graph.numItems])
        return userFinalEmbedding, itemFinalEmbedding


    def singleBehaviorPropagation(self, maxDepth, edge_index):
        layerEmbeddings = []

        edge_index = edge_index.to(self.device)

        layerEmbedding = torch.cat([user_preferences_tensor, item_ratings_tensor], dim=0).to(self.device)

        for layerNum, gcn in enumerate(self.gcn_layers):
            # print("layerNum,maxDepth:", layerNum, maxDepth)
            if layerNum >= maxDepth:
                break
            layerEmbedding = gcn(layerEmbedding, edge_index)
            # print(f"Layer {layerNum}: layerEmbedding shape: {layerEmbedding.shape}")  
            layerEmbedding = F.dropout(layerEmbedding, training=self.training)
            layerEmbeddings.append(layerEmbedding)
            # print(f"Layer {layerNum}: layerEmbeddings length: {len(layerEmbeddings)}") 

        if len(layerEmbeddings) == 0:
            raise RuntimeError("layerEmbeddings is empty. Ensure that maxDepth is greater than 0 and less than or equal to the number of GCN layers.")

        LayerEmbeddingStack = torch.stack(layerEmbeddings, dim=0)
        layerEmbeddingMean = torch.mean(LayerEmbeddingStack, dim=0)

        return layerEmbeddingMean


    def bprLoss(self, userIDs, posItemIDs, negItemIDs):

        userIDs_tensor = userIDs
        posItemIDs_tensor = posItemIDs
        negItemIDs_tensor = negItemIDs



        allUserEmbeds, allItemEmbeds = self.propagation()

        userPrefEmbeds = self.userPrefEmbeddings(user_preferences_tensor[userIDs_tensor].to(self.device))

        posItemRatingEmbeds = self.itemRatingEmbeddings(item_ratings_tensor[posItemIDs_tensor].to(self.device))
        negItemRatingEmbeds = self.itemRatingEmbeddings(item_ratings_tensor[negItemIDs_tensor].to(self.device))

        userEmbeds = allUserEmbeds[userIDs_tensor]
        posItemEmbeds = allItemEmbeds[posItemIDs_tensor]
        negItemEmbeds = allItemEmbeds[negItemIDs_tensor]

        regEmbedTerm2 = userPrefEmbeds.norm(2).pow(2) + posItemRatingEmbeds.norm(2).pow(2) + negItemRatingEmbeds.norm(2).pow(2)
        regEmbedTerm2 = regEmbedTerm2 / float(len(userIDs)) / 2.

        regEmbedTerm = userEmbeds.norm(2).pow(2) + posItemEmbeds.norm(2).pow(2) + negItemEmbeds.norm(2).pow(2)
        regEmbedTerm = regEmbedTerm / float(len(userIDs)) / 2.

        if self.attNorm == 'GAT-like':
            regEmbedTerm += self.attA.norm(2).pow(2)
            regEmbedTerm += self.attW.norm(2).pow(2) / self.attW.size()[0] / self.attW.size()[1]

            regEmbedTerm2 += self.attA.norm(2).pow(2)
            regEmbedTerm2 += self.attW.norm(2).pow(2) / self.attW.size()[0] / self.attW.size()[1]

        regBehavTerm = 0.
        for behavIterI in range(self.numLayers):
            for behavIterJ in range(behavIterI + 1, self.numLayers):

                userBehavEmbedI = self.gcn_layers[behavIterI].lin.weight
                userBehavEmbedJ = self.gcn_layers[behavIterJ].lin.weight

                cosDistanceUser = torch.sum(userBehavEmbedI * userBehavEmbedJ, dim=1)
                cosDistanceUser /= (userBehavEmbedI.norm(p=2, dim=1) * userBehavEmbedJ.norm(p=2, dim=1) + 1e-8)  
                regBehavTerm += cosDistanceUser.sum() / 3.

                posBehavEmbedI = self.gcn_layers[behavIterI].lin.weight
                posBehavEmbedJ = self.gcn_layers[behavIterJ].lin.weight
                cosDistancePos = torch.sum(posBehavEmbedI * posBehavEmbedJ, dim=1)
                cosDistancePos /= (posBehavEmbedI.norm(p=2, dim=1) * posBehavEmbedJ.norm(p=2, dim=1) + 1e-8)  
                regBehavTerm += cosDistancePos.sum() / 3.

              
                negBehavEmbedI = self.gcn_layers[behavIterI].lin.weight
                negBehavEmbedJ = self.gcn_layers[behavIterJ].lin.weight
                cosDistanceNeg = torch.sum(negBehavEmbedI * negBehavEmbedJ, dim=1)
                cosDistanceNeg /= (negBehavEmbedI.norm(p=2, dim=1) * negBehavEmbedJ.norm(p=2, dim=1) + 1e-8)  
                regBehavTerm += cosDistanceNeg.sum() / 3.

        regBehavTerm /= (float(len(userIDs_tensor)) * 2)

        posPreds = torch.mul(userEmbeds, posItemEmbeds)
        negPreds = torch.mul(userEmbeds, negItemEmbeds)

        posPreds2 = torch.mul(userPrefEmbeds, posItemRatingEmbeds)
        negPreds2 = torch.mul(userPrefEmbeds, negItemRatingEmbeds)
        # print("posPreds,posPreds2:",posPreds,posPreds2)
        # print("posPreds,posPreds2.shape:", posPreds.shape, posPreds2.shape)

        softPlus = torch.nn.Softplus()
        activeDiff = softPlus(torch.sum(negPreds, dim=1) - torch.sum(posPreds, dim=1))
        activeDiff2 = softPlus(torch.sum(negPreds2, dim=1) - torch.sum(posPreds2, dim=1))

        trueRatings = []
        for user_id, pos_item_id in zip(userIDs, posItemIDs):
            user_id_cpu = user_id.cpu().item()
            pos_item_id_cpu = pos_item_id.cpu().item()

            trueRating = user_item_ratings[
                (user_item_ratings['user_id'] == user_id_cpu) &
                (user_item_ratings['item_id'] == pos_item_id_cpu)
                ]['ratings'].values

            if len(trueRating) == 0:
                print("err")
                continue

            trueRatings.append(trueRating[0] if len(trueRating) > 0 else 0)

        trueRatings_tensor = torch.tensor(trueRatings, dtype=torch.float32).to(self.device)

        predRatings = torch.sum(userPrefEmbeds * posItemRatingEmbeds, dim=1)
        # print("predRatings:", predRatings)
        # print("predRatings.shape:", predRatings.shape)
        activeDiff3 = softPlus(F.mse_loss(predRatings, trueRatings_tensor))


        loss = torch.mean(activeDiff+activeDiff2+activeDiff3*0.1)


        return loss, regEmbedTerm,regEmbedTerm2, regBehavTerm

    # def getRatings(self,):
    def getRatings(self, userIDs: list,user_unseen_items_dict:dict):
      
        userIDs_tensor = torch.LongTensor(userIDs).to(self.device)
        finalUsers, finalItems = self.propagation()

        userRatingsEmbeds = []
        userRatingsPrefs = []
        test_items = []

        for user_id in userIDs:
            test_item = torch.LongTensor(user_unseen_items_dict[str(user_id)]).to(self.device)
            userEmbed = finalUsers[user_id]
            itemEmbed = finalItems[test_item]
            # print("test_item:", test_item)
            userRatingsEmbed = self.activeLayer(torch.matmul(userEmbed, itemEmbed.T))
            # print("userRatingsEmbed:",userRatingsEmbed.shape)
            userRatingsEmbeds.append(userRatingsEmbed)
            test_items.append(test_item)

            userPrefEmbed = self.userPrefEmbeddings(user_preferences_tensor[user_id].to(self.device))
            itemRatingEmbed = self.itemRatingEmbeddings(item_ratings_tensor[test_item].to(self.device))
            userRatingsPref = self.activeLayer(torch.matmul(userPrefEmbed, itemRatingEmbed.T))
            # print("userRatingsPref",userRatingsPref.shape)
            userRatingsPrefs.append(userRatingsPref)

        if userRatingsEmbeds and userRatingsPrefs:
            userRatingsEmbeds = torch.stack(userRatingsEmbeds)
            userRatingsPrefs = torch.stack(userRatingsPrefs)

        test_items = torch.stack(test_items)
        userRatings = userRatingsEmbeds + userRatingsPrefs*0.5
        return userRatings, test_items


    def forward(self, userIDs, itemIDs):
        userIDs_tensor = torch.LongTensor(userIDs).to(self.device)
        itemIDs_tensor = torch.LongTensor(itemIDs).to(self.device)

        user_pref_embeddings = self.userPrefEmbeddings(user_preferences_tensor[userIDs_tensor].to(self.device))
        item_rating_embeddings = self.itemRatingEmbeddings(item_ratings_tensor[itemIDs_tensor].to(self.device))
        prefBasedRatings = self.activeLayer(torch.matmul(user_pref_embeddings, item_rating_embeddings.T))


        allUserEmbeds, allItemEmbeds = self.propagation()
        userEmbeds = allUserEmbeds[userIDs_tensor]
        itemEmbeds = allItemEmbeds[itemIDs_tensor]

        embedBasedRatings = self.activeLayer(torch.matmul(userEmbeds, itemEmbeds.T))
        print("embedBasedRatings,prefBasedRatings:",embedBasedRatings,prefBasedRatings)
        print("embedBasedRatings.shape,prefBasedRatings.shape:", embedBasedRatings.shape, prefBasedRatings.shape)

        ratingPreds = embedBasedRatings + prefBasedRatings*0.5


        return ratingPreds

