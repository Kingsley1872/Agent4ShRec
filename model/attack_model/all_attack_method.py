# -*- coding: utf-8 -*-
import numpy as np
import math
import pandas as pd
from utils.api.request import get_response_from_openai
import json
from model.attack_model.Agent4Attack import FakeUserAgent, analyze_target_item
from fuzzywuzzy import process


class all_Attack:

    def __init__(self, attack_num, filler_num, n_items, target_id,
                 global_mean, global_std, item_means, item_stds, r_max, r_min, exposure_num, fixed_filler_indicator=None):
        self.attack_num = attack_num
        self.filler_num = filler_num
        self.n_items = n_items
        self.target_id = target_id
        self.global_mean = global_mean
        self.global_std = global_std
        self.item_means = item_means
        self.item_stds = item_stds
        self.r_max = r_max
        self.r_min = r_min
        self.exposure_num = exposure_num
        self.fixed_filler_indicator = fixed_filler_indicator

    def RandomAttack(self):
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        fake_profiles[:, self.target_id] = self.r_max
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def BandwagonAttack(self, selected_ids):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        fake_profiles[:, [self.target_id] + selected_ids] = self.r_max
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def AverageAttack(self):
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        fake_profiles[:, self.target_id] = self.r_max
        fn_normal = lambda iid: np.random.normal(loc=self.item_means[iid], scale=self.item_stds[iid], size=1)[0]
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = map(fn_normal, fillers)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def SegmentAttack(self, selected_ids):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        fake_profiles[:, [self.target_id] + selected_ids] = self.r_max
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            fake_profiles[i][fillers] = self.r_min
        return fake_profiles

    def MixAttack(self, bandwagon_selected, selected_ids):
        fake_profiles_random = np.zeros(shape=[self.attack_num // 4, self.n_items], dtype=float)
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles_random[:, self.target_id] = self.r_max
        for i in range(self.attack_num // 4):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles_random[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles_average = np.zeros(shape=[self.attack_num // 4, self.n_items], dtype=float)
        fake_profiles_average[:, self.target_id] = self.r_max
        fn_normal = lambda iid: np.random.normal(loc=self.item_means[iid], scale=self.item_stds[iid], size=1)[0]
        for i in range(self.attack_num // 4):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = map(fn_normal, fillers)
            for f_id, r in zip(fillers, ratings):
                fake_profiles_average[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles_bandwagon = np.zeros(shape=[self.attack_num // 4, self.n_items], dtype=float)
        fake_profiles_bandwagon[:, [self.target_id] + selected_ids] = self.r_max
        for i in range(self.attack_num // 4):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles_bandwagon[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles_segment = np.zeros(shape=[self.attack_num - self.attack_num // 4 * 3, self.n_items], dtype=float)
        fake_profiles_segment[:, [self.target_id] + selected_ids] = self.r_max
        for i in range(self.attack_num - self.attack_num // 4 * 4):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            fake_profiles_segment[i][fillers] = self.r_min
        fake_profiles = np.concatenate([
                fake_profiles_random,
                fake_profiles_average,
                fake_profiles_bandwagon,
                fake_profiles_segment
            ], 
            axis=0
        )
        return fake_profiles

    def AoPAttack(self, dataset_name):
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        fake_profiles[:, self.target_id] = self.r_max
        fn_normal = lambda iid: np.random.normal(loc=self.item_means[iid], scale=self.item_stds[iid], size=1)[0]
        for i in range(self.attack_num):
            item_popularity = pd.read_csv(f"AgentAttack/data/{dataset_name}/item_popularity.csv", encoding='utf-8')
            top_10_percent_items = item_popularity.nlargest(int(len(item_popularity) * 0.1), 'popularity')['item_id'].values
            top_10_percent_items = list(set(top_10_percent_items) - {self.target_id})
            fillers = np.random.choice(top_10_percent_items, size=self.filler_num, replace=False)
            ratings = map(fn_normal, fillers)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def get_target_item_info(self, item_popularity, dataset_name):
        u_item_df = pd.read_csv(f"AgentAttack/data/{dataset_name}/u_item.csv", encoding='utf-8')
        target_item = u_item_df[u_item_df['item_id'] == self.target_id]
        target_item_name = target_item['title'].values[0]
        if dataset_name == "ml100k":
            target_item_type = target_item['genre'].values[0]
        else:
            target_item_type = target_item['categories'].values[0]
        target_item_mean = self.item_means[self.target_id]
        target_item_popularity = item_popularity[item_popularity['item_id'] == self.target_id]['popularity'].values[0]
        med_popularity = item_popularity['popularity'].median()
        target_item_popularity = "high" if target_item_popularity > med_popularity else "low"
        return target_item_name, target_item_type, target_item_mean, target_item_popularity

    def LLMAttack(self, dataset_name):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id]))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        fake_profiles[:, self.target_id] = self.r_max
        fake_reviews = {}
        item_popularity = pd.read_csv(f"AgentAttack/data/{dataset_name}/item_popularity.csv", encoding='utf-8')
        target_item_name, target_item_type, target_item_mean, target_item_popularity = self.get_target_item_info(item_popularity, dataset_name)
        target_item_analysis = analyze_target_item(target_item_name, target_item_type, target_item_popularity, dataset_name)
        for i in range(self.attack_num):
            attacker = FakeUserAgent("", "")
            attacker.get_fake_profile(target_item_name, target_item_type, target_item_mean, target_item_popularity, target_item_analysis, dataset_name)
            target_review_dict = attacker.review_target(target_info=target_item_analysis, target_item_name=target_item_name, target_item_type=target_item_type)
            all_reviews = {}
            all_reviews[self.target_id] = list(target_review_dict.values())[0]
            p = [item_popularity[item_popularity['item_id'] == item]['popularity'].values[0] if item in item_popularity['item_id'].values else 1 for item in filler_candis]
            exposurers = np.random.choice(filler_candis, size=10 * self.exposure_num, replace=False, p=p / sum(p))
            ratings, reviews = self.get_fake_action(attacker, exposurers, target_item_analysis, dataset_name)
            all_reviews.update(reviews)
            for f_id, r in zip(exposurers, ratings):
                if r == -1:
                    continue
                r = max(1, min(5, r))
                fake_profiles[i][f_id] = r
            fake_reviews[str(i)] = all_reviews
            print(f"fake user {i} completed.")
        return fake_profiles, fake_reviews

    def get_fake_action(self, attacker, exposurers, target_info, dataset_name):
        u_item_df = pd.read_csv(f"AgentAttack/data/{dataset_name}/u_item.csv", encoding='utf-8')
        exposurers_info = u_item_df[u_item_df['item_id'].isin(exposurers)]
        exposurers_name = exposurers_info['title'].values
        if dataset_name == "ml100k":
            exposurers_type = exposurers_info['genre'].values
        else:
            exposurers_type = exposurers_info['categories'].values
        exposurers_mean = [round(self.item_means[filler], 1) for filler in exposurers]
        all_ratings = []
        all_reviews = {}
        for i in range(0, len(exposurers_name), 10):
            exposure_item_info = ""
            for j in range(i, min(i + 10, len(exposurers_name))):
                if dataset_name == "ml100k":
                    exposure_item_info += f"Movie name: {exposurers_name[j]} (genre: {exposurers_type[j]}, rated: {exposurers_mean[j]})\n"
                else:
                    exposure_item_info += f"Product name: {exposurers_name[j]} (genre: {exposurers_type[j]}, rated: {exposurers_mean[j]})\n"
            rating_dict = attacker.perform_action(exposure_item_info=exposure_item_info, rating_dict=None, action="rating", target_info=target_info, dataset_name=dataset_name)
            ratings = self.get_rating_vector(rating_dict, exposurers_name[i: min(i + 10, len(exposurers_name))])
            all_ratings.extend(ratings)
            review_dict = attacker.perform_action(exposure_item_info=exposure_item_info, rating_dict=rating_dict, action="review", target_info=target_info, dataset_name=dataset_name)
            review_dict = self.get_review_id(review_dict, exposurers_name[i: min(i + 10, len(exposurers_name))], exposurers_info[i: min(i + 10, len(exposurers_name))])
            all_reviews.update(review_dict)
            attacker.add_memory(rating_dict, mode="rating")
            attacker.add_memory(review_dict, mode="review")
        return all_ratings, all_reviews

    def get_rating_vector(self, rating_dict, exposurers_name):
        ratings = [-1] * len(exposurers_name)
        for key in rating_dict.keys():
            best_match = process.extractOne(key, exposurers_name)
            if best_match:
                index = exposurers_name.tolist().index(best_match[0])
                ratings[index] = rating_dict[key]
        return ratings

    def get_review_id(self, review_dict, exposurers_name, exposurers_info):
        updated_review_dict = {}
        for key in review_dict.keys():
            best_match = process.extractOne(key, exposurers_name)
            if best_match:
                index = exposurers_name.tolist().index(best_match[0])
                item_id = exposurers_info.iloc[index]['item_id']
                updated_review_dict[item_id] = review_dict[key]
        return updated_review_dict
