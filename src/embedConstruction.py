from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import numpy as np
import pandas as pd


def predict_test(model, device, filtered_data):
    candidate_list = ["positive", "neutral", "negative"]
    term_list = ["place", "food", "service", "staff", "miscellaneous", "price"]

    model.eval()
    model.config.use_cache = False

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    user_term_counts = pd.DataFrame(0, index=filtered_data['user_id'].unique(), columns=term_list)
    item_term_counts = pd.DataFrame(0, index=filtered_data['item_id'].unique(), columns=term_list)
    total = len(filtered_data)
    count = 0

    for _, row in filtered_data.iterrows():
        count += 1
        text = row['reviews']
        item_id = row['item_id']
        user_id = row['user_id']
        best_term = None
        best_prob = -1
        best_sentiment_idx = 0

        for term in term_list:
            target_list = [f"The sentiment polarity of {term.lower()} is {candi.lower()}." for candi in candidate_list]

            input_ids = tokenizer([text] * 3, return_tensors='pt')['input_ids']
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']

            with torch.no_grad():
                output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
                logits = output.softmax(dim=-1).to('cpu').numpy()

            score_list = []
            for i in range(3):
                score = 1
                for j in range(logits[i].shape[0] - 2):
                    score *= logits[i][j][output_ids[i][j + 1]]
                score_list.append(score)

            max_score = max(score_list)
            if max_score > best_prob:
                best_prob = max_score
                best_sentiment_idx = np.argmax(score_list)
                best_term = term

        polarity = candidate_list[best_sentiment_idx]

        if polarity == "positive":
            user_term_counts.loc[user_id, best_term] += 1
            item_term_counts.loc[item_id, best_term] += 1
        elif polarity == "negative":
            user_term_counts.loc[user_id, best_term] += 2
            item_term_counts.loc[item_id, best_term] -= 1
        else:
            user_term_counts.loc[user_id, best_term] += 0.5

    return user_term_counts, item_term_counts

