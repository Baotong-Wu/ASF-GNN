import numpy as np
import scipy.sparse as sp
from os.path import join

import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, coo_matrix
import pandas as pd
import random
import json

import parse
from utils import enPrint
from parse import argParser
from sklearn.preprocessing import MinMaxScaler

class GraphDataset(Dataset):
    """
        Data Loader
        IN THE FORM OF **GRAPH**
    """

    def __init__(self, datasetName="Amazon-CDs"):
        super(GraphDataset, self).__init__()
        self.datasetName = datasetName
        self.filePath = None
        if self.datasetName == 'Amazon-Sports':
            self.filePath = '../data/Amazon-sports'
        elif self.datasetName == 'Amazon-CDs':
            self.filePath = '../data/Amazon-cds'
        elif self.datasetName == 'Amazon-Health':
            self.filePath = '../data/Amazon-Health'
        elif self.datasetName == 'Yelp':
            self.filePath = '../data/Yelp'

        self.numUsers = 0
        self.numItems = 0
        self.trainInteractions = 0
        self.testInteractions = 0
        self.normGraph = None
        self.testSampleDict = {}

        with open(join(self.filePath, "train.txt"), "r") as trainFile:
            enPrint("Training Dataset Loading...")
            # trainRowIndices: userID list
            trainRowIndices = []
            # trainColIndices: itemID list
            trainColIndices = []
            for line in trainFile.readlines():
                if len(line) >= 0:
                    userItems = line.strip().split(" ")
                    userID = int(userItems[0])
                    itemIDs = [int(itemIDStr) for itemIDStr in userItems[1:]]
                    trainRowIndices.extend([userID] * len(itemIDs))
                    trainColIndices.extend(itemIDs)
                    # Number of Users, denoted as M
                    self.numUsers = max(userID, self.numUsers)
                    # Number of Items, denoted as N
                    self.numItems = max(max(itemIDs), self.numItems)
                    self.trainInteractions += len(itemIDs)

        with open(join(self.filePath, "test.txt"), "r") as testFile:
            enPrint("Testing Dataset Loading...")
            testRowIndices = []
            testColIndices = []
            for line in testFile.readlines():
                if len(line) >= 0:
                    userItems = line.strip().split(" ")
                    userID = int(userItems[0])
                    itemIDs = [int(itemIDStr) for itemIDStr in userItems[1:]]
                    testRowIndices.extend([userID] * len(itemIDs))
                    testColIndices.extend(itemIDs)
                    self.numUsers = max(userID, self.numUsers)
                    self.numItems = max(max(itemIDs), self.numItems)
                    self.testInteractions += len(itemIDs)
                    self.testSampleDict[userID] = itemIDs

            # userID、itemID start from 0
            self.numUsers += 1
            self.numItems += 1
            print(f"{len(trainRowIndices)} Interactions in Training Dataset")
            print(f"{len(testRowIndices)} Interactions in Testing Dataset")
            print(f"{self.datasetName} Sparsity:"
                  f"{(len(trainRowIndices) + len(testRowIndices)) / self.numUsers / self.numItems}")
            data = np.ones_like(trainRowIndices)
            # self.adjMatrix: adjacency matrix on traning set
            self.adjMatrix = csr_matrix((data, (trainRowIndices, trainColIndices)),
                                        shape=(self.numUsers, self.numItems))
            self.userDArray = np.array(self.adjMatrix.sum(axis=0)).squeeze()
            self.userDArray[self.userDArray == 0.] = 1
            self.itemDArray = np.array(self.adjMatrix.sum(axis=1)).squeeze()
            self.itemDArray[self.itemDArray == 0.] = 1

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    @property
    def getUserPosItems(self):
        """
        Access positive item ID list of each user in traning set
        :return: list of lists of positive items
        """
        posItemIDsList = []
        for userID in range(self.numUsers):
            _, posItemIDs = self.adjMatrix[userID].nonzero()
            posItemIDsList.append(posItemIDs)
        return posItemIDsList

    @property
    def getNormAdj(self):
        """
        Build the Normalized Adjacency Matrix.
            A =
            :math:`|0,     R|`\n
            :math:`|R^T,   0|`
        :return: csr_matrix
        """
        # enPrint(f"Normalized Adjacency Matrix Loading...")
        # npzFileName = "s_pre_adj_mat.npz"
        npzFileName = "normAdjMatrix.npz"

        def transCsrMatrix2SparseTensor(csrMatrix: csr_matrix) -> torch.sparse.FloatTensor:
            """
            Convert CSR_Matrix to Torch Sparse Float Tensor
            :param csrMatrix: CSR_Matrix
            :return: Torch Sparse Float Tensor
            """
            cooMatrix: coo_matrix = csrMatrix.tocoo()
            matrixTensor = torch.sparse.FloatTensor(torch.LongTensor([cooMatrix.row.tolist(), cooMatrix.col.tolist()]),
                                                    torch.FloatTensor(cooMatrix.data.astype(np.float32)))
            return matrixTensor

        if self.normGraph is None:
            try:
                normG = sp.load_npz(join(self.filePath, npzFileName))
                enPrint("NPZ Matrix Loaded Successfully...")
            except FileNotFoundError:
                enPrint("Generating Normalized Adjacency Matrix from Scratch...")
                normG = sp.dok_matrix((self.numUsers + self.numItems, self.numItems + self.numUsers))
                normG = normG.tolil()
                R = self.adjMatrix.tolil()
                normG[:self.numUsers, self.numUsers:] = R
                normG[self.numUsers:, :self.numUsers] = R.T
                nodeDegrees = np.array(normG.sum(axis=1)).squeeze()
                nodeDegreeSqrts = np.power(nodeDegrees, -0.5)
                nodeDegreeSqrts[np.isinf(nodeDegreeSqrts)] = 0.
                diagMatrix = sp.diags(nodeDegreeSqrts)
                normG = diagMatrix.dot(normG).dot(diagMatrix)
                normG = normG.tocsr()
                sp.save_npz(join(self.filePath, npzFileName), normG)

            self.normGraph = transCsrMatrix2SparseTensor(normG)
            if parse.CUDA_AVAILABLE:
                self.normGraph = self.normGraph.coalesce().to(parse.DEVICE)

        return self.normGraph

    def getBPRSamples(self):
        """
        Sampling of triplets - User, Positive Sample, Negtive Sample
        :math:`[[sampleUserID_0, posItemID_0, negItemID_0]`,
        :math:`\cdots,`
        :math:`[sampleUserID_n, posItemID_n, negItemID_n]]`
        :return: samples
        """
        samples = []
        posItemIDsList = self.getUserPosItems
        sampleUserIDs = np.random.randint(0, self.numUsers, self.trainInteractions)
        for sampleUserID in sampleUserIDs:
            posItemIDs4User = posItemIDsList[sampleUserID]
            posItemID = posItemIDs4User[np.random.randint(0, len(posItemIDs4User))]
            # Negative Item Sampling for **sampleUserID**
            while True:
                negItemID = np.random.randint(0, self.numItems)
                if negItemID not in posItemIDs4User:
                    break
            samples.append([sampleUserID, posItemID, negItemID])
        return np.array(samples)


import pandas as pd
import random
import json



def data_preprocess(dataset_name):
    """
    Processes the input data file to extract user-item-rating information and save it to a CSV file.
    :param dataset_name: Name of the dataset ('Yelp' or 'Amazon-CDs'...).
    """
    base_path = f'./data/{dataset_name}/'

    # Path to the raw input CSV file
    data_file_path = base_path + f'{dataset_name.lower()}_review_data.csv'

    # Read the CSV file
    data = pd.read_csv(data_file_path)

    # Filter relevant columns (user_id, item_id, ratings)
    filtered_data = data[['user_id', 'item_id', 'ratings']]
    filtered_data.to_csv(base_path + 'user_item_ratings.csv', index=False)

    generate_train_test_data(base_path)
    return base_path


def generate_train_test_data(base_path):
    """
    Processes the user-item rating data to generate the training set, test set, and final test data.
    :param base_path: Path to the dataset directory (e.g., './data/Yelp/' or './data/Amazon/').
    """
    input_file = base_path + 'user_item_ratings.csv'

    data = pd.read_csv(input_file)

    all_users = data['user_id'].unique()
    all_items = set(data['item_id'].unique())

    top2_data = data.groupby('user_id').apply(lambda x: x.nlargest(2, 'ratings')).reset_index(drop=True)

    # Initialize the test data list
    test_data_list = []

    for user_id in top2_data['user_id'].unique():
        user_top2_items = top2_data[top2_data['user_id'] == user_id]
        random_item = user_top2_items.sample(1)
        test_data_list.append(random_item)

    test_data = pd.concat(test_data_list, ignore_index=True)

    train_data = data.drop(test_data.index)
    test_data = test_data.drop(columns=['ratings'])
    train_data = train_data.drop(columns=['ratings'])
    save_to_file(test_data, base_path + 'test.txt')
    save_to_file(train_data, base_path + 'train.txt')
    final_test_data = {}

    for user_id in test_data['user_id']:
        # Get the items the user has interacted with
        interacted_items = set(data[data['user_id'] == user_id]['item_id'].tolist())

        # Get the items that the user has not interacted with
        non_interacted_items = list(all_items - interacted_items)
        selected_non_interacted_items = random.sample(non_interacted_items, 99)

        # Get the actual item the user interacted with (highest or second-highest rating)
        true_interaction = test_data[test_data['user_id'] == user_id].iloc[0]['item_id']

        # Add the selected non-interacted items along with the true interaction item
        final_test_data[str(user_id)] = selected_non_interacted_items + [int(true_interaction)]

    # Save the final test data as a JSON file
    with open(base_path + 'user_unseen_items_dict.json', 'w') as f:
        json.dump(final_test_data, f, indent=4)


# Helper function to save user-item interactions to a file
def save_to_file(data, filename):
    """
    Save user-item interaction data to a file.
    :param data: DataFrame containing user-item interactions.
    :param filename: File path to save the data.
    """
    with open(filename, 'w') as file:
        file.writelines([f"{user_id} {' '.join(map(str, group['item_id'].tolist()))}\n"
                         for user_id, group in data.groupby('user_id')])




def process_data(user_preferences, item_ratings, user_item_ratings):
    """
    Process the raw data into tensors for model training.
    Returns:
        user_preferences_tensor: Normalized user preferences as torch tensor.
        item_ratings_tensor: Normalized item ratings as torch tensor.
        user_ids: List of user IDs.
        item_ids: List of item IDs.
        user_id_to_index: Dictionary mapping user ID to index.
        item_id_to_index: Dictionary mapping item ID to index.
        rating_matrix: The user-item interaction matrix.
        ideal_item_embeddings_tensor: Tensor of ideal item embeddings.
        edge_index: Edge index tensor for graph construction.
    """

    # Normalize user preferences and item ratings data
    scaler = MinMaxScaler(feature_range=(0, 1))
    user_preferences_normalized = scaler.fit_transform(user_preferences.iloc[:, 1:])
    item_ratings_normalized = scaler.fit_transform(item_ratings.iloc[:, 1:])

    user_preferences_tensor = torch.tensor(user_preferences_normalized, dtype=torch.float32)
    item_ratings_tensor = torch.tensor(item_ratings_normalized, dtype=torch.float32)

    user_ids = user_preferences['user_id'].tolist()
    item_ids = item_ratings['item_id'].tolist()
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    num_users = len(user_ids)
    num_items = len(item_ids)
    rating_matrix = torch.zeros(num_users, num_items)

    user_ideal_items = {}
    for index, row in user_item_ratings.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['ratings']
        if rating > 4:
            if user_id not in user_ideal_items:
                user_ideal_items[user_id] = []
            user_ideal_items[user_id].append(item_id)

    ideal_item_embeddings_tensor = torch.zeros(num_users, item_ratings_tensor.shape[1], dtype=torch.float32)
    for user_id, item_list in user_ideal_items.items():
        item_indices = [item_id_to_index[item_id] for item_id in item_list if item_id in item_id_to_index]
        if len(item_indices) > 0:
            item_embeddings = item_ratings_tensor[item_indices]
            ideal_item_embed = torch.mean(item_embeddings, dim=0)
            if user_id in user_id_to_index:
                user_index = user_id_to_index[user_id]
                ideal_item_embeddings_tensor[user_index] = ideal_item_embed

    user_indices = torch.LongTensor(user_item_ratings['user_id'].values)
    item_indices = torch.LongTensor(user_item_ratings['item_id'].values)
    user_item_edge_index = []
    for user_id, item_id in zip(user_indices, item_indices):
        user_index = user_id_to_index[user_id.item()]
        item_index = item_id_to_index[item_id.item()] + num_users
        user_item_edge_index.append([user_index, item_index])
        user_item_edge_index.append([item_index, user_index])

    edge_index = torch.tensor(user_item_edge_index, dtype=torch.long).t().contiguous()

    return user_preferences_tensor, item_ratings_tensor, user_ids, item_ids, user_id_to_index, item_id_to_index, rating_matrix, ideal_item_embeddings_tensor, edge_index

