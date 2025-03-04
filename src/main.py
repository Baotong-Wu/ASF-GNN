import torch
import time
import json
import numpy as np
import random
from collections import defaultdict
from utils import miniBatch
from torch.optim import Adam
from warnings import simplefilter
from tensorboardX import SummaryWriter
from parse import argParser
from parse import CUDA_AVAILABLE
from parse import DEVICE
from utils import enPrint, setSeed, evaluate_hr_ndcg
from transformers import BartForConditionalGeneration, BartTokenizer
from model import ASFGNN
from dataProcessor import GraphDataset, data_preprocess, process_data
from embedConstruction import  predict_test
from utils import shuffle
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Model Parameter Configuration
args = argParser()
BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.test_batch_size
LR = args.lr
EPOCHS = args.epochs
SEED = args.seed
PREFIX = args.prefix
DATASET = args.dataset
TOPK = args.topK
WEIGHT_DECAY1 = args.weight_decay_embed
WEIGHT_DECAY2 = args.weight_decay_behavior
FINAL_INTEGRATION = args.final_integration
IF_REG_BEHAV = args.ifRegBehav
IF_REG_EMBEDDING = args.ifRegEmbedding
IF_DROPOUT = args.ifDropOut
IF_LOAD = args.ifLoad
LOAD_MODEL_NAME = args.load_model_name

# Preparation
setSeed(SEED)
simplefilter(action="ignore", category=FutureWarning)

# Environment Configuration
ROOT_PATH = "/".join(os.path.abspath(__file__).split("/")[:-2])  # for Linux Environment
# ROOT_PATH = "\\".join(os.path.abspath(__file__).split("\\")[:-2])  # for Dos Environment
LOG_PATH = os.path.join(ROOT_PATH, "log")
DATA_PATH = os.path.join(ROOT_PATH, "data")
BOARD_PATH = os.path.join(LOG_PATH, "runs")
CHECKPOINT_PATH = os.path.join(LOG_PATH, "checkpoints")
MODEL_DUMP_FILENAME = os.path.join(CHECKPOINT_PATH, LOAD_MODEL_NAME)
DUMP_FILE_PREFIX = MODEL_DUMP_FILENAME.split("epoch")[0]
DUMP_FILE_SUFFIX = ".pth.tar"

# TensorBoard
writer = SummaryWriter(logdir=os.path.join(BOARD_PATH,
                        args.prefix + time.strftime("%Y%m%d-%Hh%Mm%Ss",
                        time.localtime(time.time()))),
                       comment=PREFIX)



def generate_test_set(test_sample_data, graph, num_items, batch_size=1024):
    """
    Generate test set for each user
    :param test_sample_data: dict-type variable，key - userID， value - according item list
    :param graph: Graph Dataset
    :param num_items: total number of items
    :param batch_size: batch size for processing
    :return: user_unseen_items_dict
    """
    all_item_ids_set = set(map(int, np.arange(num_items)))
    user_unseen_items_dict = defaultdict(list)
    test_user_ids = list(test_sample_data.keys())

    for batch_user_ids in miniBatch(test_user_ids, batchSize=10240):
        for user_id in batch_user_ids:
            user_true_items = set(map(int, test_sample_data[user_id]))
            user_true_items_sample = random.sample(user_true_items, 1)
            user_train_items = set(map(int, graph.getUserPosItems[user_id]))  # 用户在训练集中的正向交互物品
            user_unseen_items = list(all_item_ids_set - user_train_items-user_true_items)

            if len(user_unseen_items) >= 99:
                user_unseen_sample = random.sample(user_unseen_items, 99)
            else:
                user_unseen_sample = user_unseen_items

            user_test_items = user_unseen_sample + list(user_true_items_sample)
            user_unseen_items_dict[int(user_id)] = list(map(int, user_test_items))

    with open("../data/Yelp/user_unseen_items_dict_2.json", 'w') as f:
        json.dump(user_unseen_items_dict, f)

    return user_unseen_items_dict



# Train
def train(userIDs, posItemIDs, negItemIDs, epoch):

    global globalTrainStep
    model.train()
    userIDs, posItemIDs, negItemIDs = shuffle(userIDs, posItemIDs, negItemIDs)
    batchIterNum = len(userIDs) // BATCH_SIZE + 1
    averageLoss = 0.
    for (batchIter, (batchUser, batchPos, batchNeg)) in enumerate(
            miniBatch(userIDs, posItemIDs, negItemIDs, batchSize=BATCH_SIZE)):

        # Ensure tensors are on the correct device
        if CUDA_AVAILABLE:
            batchUser = batchUser.to(DEVICE)
            batchPos = batchPos.to(DEVICE)
            batchNeg = batchNeg.to(DEVICE)

        loss, regEmbedTerm, regEmbedTerm2,regBehavTerm = model.bprLoss(userIDs=batchUser, posItemIDs=batchPos, negItemIDs=batchNeg)
        print(f"\tLOSS:{loss:.6f}\tREG1:{regEmbedTerm:.6f}\tREG2:{regBehavTerm:.6f}\tREG3:{regEmbedTerm2:.6f}", end='\t')
        loss += regEmbedTerm * 1e-3 + regBehavTerm * 1e-4 + regEmbedTerm2 * 1e-8
        print(f"Total LOSS:{loss:.6f}")

        # Gradient propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tensorboard Writing
        averageLoss += loss.item()
        globalTrainStep += 1
        writer.add_scalar("Train/BPR_LOSS", loss.item(), global_step=globalTrainStep, walltime=time.time())

    averageLoss /= batchIterNum
    print(f"[EPOCH {epoch:4d}] - [LOSS]: {averageLoss: .6f}")


# Test
def test(testSampleData: dict, globalStep: int,user_unseen_items_dict:dict):
    """
    Test and Metric Calculation
    :param testSampleData: dict-type variable，key - userID， value - according item list
    :param globalStep: global mini-batch step，used to record metric changes
    :return: None
    """
    testUserIDs = list(testSampleData.keys())
    maxTopK = max(TOPK)
    metrics = np.zeros((4, len(TOPK)))
    model.eval()

    with torch.no_grad():
        batchRecalls = []
        batchPrecisions = []
        batchNDCGs = []
        batchHRs = []
        results = []
        best_result = []
        for batchUserIDs in miniBatch(testUserIDs, batchSize=TEST_BATCH_SIZE):
            groundTruePosItems = [testSampleData[userID] for userID in batchUserIDs]
            testRatings, test_items = model.getRatings(batchUserIDs,user_unseen_items_dict)
            testRatings = -testRatings
            rank = testRatings.argsort().argsort()[:, -1]
            hr_20, ndcg_20, hr_10, ndcg_10, hr_5, ndcg_5 = evaluate_hr_ndcg(rank)

            metrics2 = [
                [hr_5, hr_10, hr_20],
                [ndcg_5, ndcg_10, ndcg_20]
            ]

            writer.add_scalars(f'Test2/HR',
                               {'HR@' + str(TOPK[i]): metrics2[0][i] for i in range(len(TOPK))}, globalStep)
            writer.add_scalars(f'Test2/NDCG',
                               {'NDCG@' + str(TOPK[i]): metrics2[1][i] for i in range(len(TOPK))}, globalStep)
            # 打印指标
            enPrint(f"[TEST]")
            for k in range(len(TOPK)):
                print(f"HR@{TOPK[k]:2d}: {metrics2[0][k]: .6f}", end='\t')
                print(f"NDCG@{TOPK[k]:2d}: {metrics2[1][k]: .6f}", end='\t')
                print()



def trainTripleSampling(graph: GraphDataset):
    """
    Training Triplet Initialization
    :param graph: Graph Dataset
    :return: Randomly Sampled Triplets
    """
    trainTripleData = graph.getBPRSamples()
    userID = trainTripleData[:, 0]
    posItemID = trainTripleData[:, 1]
    negItemID = trainTripleData[:, 2]

    userID = torch.from_numpy(userID).long()
    posItemID = torch.from_numpy(posItemID).long()
    negItemID = torch.from_numpy(negItemID).long()

    if CUDA_AVAILABLE:
        userID = userID.to(DEVICE)
        posItemID = posItemID.to(DEVICE)
        negItemID = negItemID.to(DEVICE)
    return userID, posItemID, negItemID


if __name__ == '__main__':
    base_path = data_preprocess(DATASET)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model2 = BartForConditionalGeneration.from_pretrained('./outputs').to(device)

    # Load data
    review_data = pd.read_csv(base_path + f'processed_{DATASET.lower()}_data.csv')

    # embedding construction
    user_preferences, item_ratings = predict_test(model2, device, review_data)
    user_item_ratings = pd.read_csv(base_path + f'rating.csv')
    process_data(user_preferences, item_ratings, user_item_ratings)
    graph = GraphDataset(DATASET)
    print("graph.shape:",graph.getNormAdj.shape)
    test_sample_data = graph.testSampleDict
    model = ASFGNN(graph=graph)
    if CUDA_AVAILABLE:
        model = model.to(DEVICE)

    user_unseen_items_dict = {}
    num_items = 38964

    with open("../data/Yelp/user_unseen_items_dict_1.json", 'r') as f:
        user_unseen_items_dict = json.load(f)
        # keys = list(user_unseen_items_dict.keys())
        # print("Keys in user_unseen_items_dict:", keys)

    if IF_LOAD:
        try:
            model.load_state_dict(torch.load(MODEL_DUMP_FILENAME), strict=False)
            enPrint("Model Loaded from Dump File...")
            test(test_sample_data, int(LOAD_MODEL_NAME.split(".")[0].split("-")[-1]), user_unseen_items_dict)
            exit(0)
        except FileNotFoundError as exp:
            print(MODEL_DUMP_FILENAME + " NOT FOUND!")
        finally:
            writer.close()

    optimizer = Adam(model.parameters(), lr=LR)
    average_loss = 0.
    globalTrainStep = 0

    try:
        for epoch in range(EPOCHS):
            if epoch % 10 == 0:
                test(test_sample_data, epoch, user_unseen_items_dict)
                torch.save(model.state_dict(), DUMP_FILE_PREFIX + "epoch-" + str(epoch) + DUMP_FILE_SUFFIX)
            user_id, pos_item_id, neg_item_id = trainTripleSampling(graph)
            if CUDA_AVAILABLE:
                user_id, pos_item_id, neg_item_id = user_id.to(DEVICE), pos_item_id.to(DEVICE), neg_item_id.to(DEVICE)
            train(user_id, pos_item_id, neg_item_id, epoch)
    finally:
        writer.close()

