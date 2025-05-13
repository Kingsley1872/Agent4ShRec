import os
from utils.api.request import get_response_from_openai
from fuzzywuzzy import process
from time import sleep
import numpy as np

os.environ["OPENAI_API_KEY"] = ""

def get_embedding_cosine_similarity(text):
    from utils.api.request import get_embedding_from_openai
    embedding = get_embedding_from_openai(text, model="text-embedding-3-small")
    return np.array(embedding)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-8)

def analyze_target_item(target_item_name, target_item_type, target_item_popularity, dataset_name):
    if dataset_name == "ml100k":
        prompt = f"Analyze the movie {target_item_name} ({target_item_type} genre, {target_item_popularity} popularity). Provide a comprehensive analysis of its characteristics and output 10 or more short phrases describing the movie from different perspectives, such as genre, theme, style, or director's approach. Output the result directly as a string list without other words, e.g., [\"Visually stunning scenes\", \"Hilarious and entertaining plot\", ...]."
    else:
        prompt = f"Analyze the product {target_item_name} ({target_item_type} genre, {target_item_popularity} popularity). Provide a comprehensive analysis of its characteristics and output 10 or more short phrases describing the product from different perspectives, such as genre, price, style, or store. Output the result directly as a string list without other words, e.g., [\"Visually stunning scenes\", \"Entertaining function\", ...]."
    responseText = None
    for _ in range(10):
        try:
            responseText = get_response_from_openai(prompt, model="gpt-4o-mini", temperature=0.7)
            target_analysis_list = eval(responseText)
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            responseText = None
            sleep(10)
    if responseText is None:
        raise ValueError("Failed to get a valid response after 10 attempts.")
    return target_analysis_list

class FakeUserAgent:
    def __init__(self, profile, action):
        self.profile = profile
        self.memory = {"rating": {}, "review": {}}
        self.action = action

    def validate_profile(self):
        profile = self.profile.lower()
        required_keywords = ["age", "occupation", "like", "dislike", "gender"]
        issues = []

        if not any(k in profile for k in required_keywords):
            issues.append("Missing required persona attributes like age, occupation, or preferences.")
        if len(self.profile) > 100:
            issues.append("Self-introduction exceeds reasonable length (should be concise).")
        if "i love movies" in profile or "i like movies" in profile:
            issues.append("Profile contains overly generic phrases.")

        return issues if issues else None

    def check_profile_diversity(self, new_profile, existing_profiles, threshold=0.9):
        new_emb = get_embedding_cosine_similarity(new_profile)
        for profile in existing_profiles:
            emb = get_embedding_cosine_similarity(profile)
            similarity = cosine_similarity(new_emb, emb)
            if similarity > threshold:
                return False
        return True
    
    def get_fake_profile(self, target_item_name, target_item_type, target_item_mean, target_item_popularity, target_analysis_list, dataset_name, existing_profiles=[]):
        for _ in range(10):
            if dataset_name == "ml100k":
                prompt = (
                    f"Act as a movie enthusiast whose favorite film is {target_item_name} "
                    f"({target_item_type} genre, {round(target_item_mean, 1)}/5 avg rating, {target_item_popularity} popularity). "
                    f"Keywords describing the film: {target_analysis_list}. "
                    f"Based on these, create a persona including: 1. Profile: gender, age, occupation, film preferences (likes/dislikes). "
                    f"2. Behavior: rating/review habits, genre exploration, trend-following. "
                    f"Output a self-introduction (max 80 words) starting with 'Here is my self-introduction:' "
                    f"using concise, specific language with unique details."
                )
            else:
                prompt = (
                    f"Act as an amazon buyer whose favorite product is {target_item_name} "
                    f"({target_item_type} genre, {round(target_item_mean, 1)}/5 avg rating, {target_item_popularity} popularity). "
                    f"Keywords describing the product: {target_analysis_list}. "
                    f"Based on these, create a persona including: 1. Profile: gender, age, occupation, product preferences (likes/dislikes). "
                    f"2. Behavior: rating/review habits, genre exploration, trend-following. "
                    f"Output a self-introduction (max 80 words) starting with 'Here is my self-introduction:' "
                    f"using concise, specific language with unique details."
                )

            try:
                response = get_response_from_openai(prompt, model="gpt-4o-mini", temperature=1.5)
                fake_feature = response.split("Here is my self-introduction: ")[-1].strip()
                self.profile = fake_feature

                if self.validate_profile():
                    continue

                if not self.check_profile_diversity(self.profile, existing_profiles):
                    continue

                break
            except Exception as e:
                print(f"Error during profile generation: {e}")
                sleep(10)

    def add_memory(self, new_memory, mode):
        if mode == "rating":
            self.memory["rating"].update(new_memory)
        elif mode == "review":
            self.memory["review"].update(new_memory)

    def retrieve_recent_memory(self, max_entries=3):
        rating_memory = self.memory.get("rating", {})
        review_memory = self.memory.get("review", {})

        combined_memory = []

        for movie, rating in rating_memory.items():
            combined_memory.append((movie, f"rated {movie} {rating}/5"))

        for movie, review in review_memory.items():
            combined_memory.append((movie, f"reviewed {movie}: {review}"))

        recent_entries = combined_memory[-max_entries:]

        if recent_entries:
            return "Here are some of my most recent memories:\n" + "\n".join(f"- I {entry[1]}" for entry in recent_entries)
        else:
            return None

    def retrieve_memory(self, exposure_item_info):
        movie_names = [movie_info.split(" (")[0].split("Movie name:")[-1].strip() for movie_info in exposure_item_info.split("\n")]
        rating_memory = self.memory["rating"]
        review_memory = self.memory["review"]
        matched_movie_names = process.extractOne(" ".join(movie_names), list(rating_memory.keys()) + list(review_memory.keys()))
        if matched_movie_names is not None:
            matched_movie_name = matched_movie_names[0]
            if matched_movie_name in rating_memory:
                return f"Based on my memory, I rated {matched_movie_name} {rating_memory[matched_movie_name]}/5."
            if matched_movie_name in review_memory:
                return f"Based on my memory, I reviewed {matched_movie_name}: {review_memory[matched_movie_name]}."
        return None
    
    def review_target(self, target_info, target_item_name, target_item_type):
        prompt = f"This is your self-introduction: {self.profile}.\nWrite a comment about your experience with the product {target_item_name} ({target_item_type} type), expressing your appreciation naturally within 80 words. Use the following keywords as a reference: {target_info}. Keep the style consistent with your background.\nOutput the result directly in dictionary format, e.g., {{\"item_name\": \"I really enjoyed this product for its excellent quality!\"}}."
        responseText = None
        for _ in range(10):
            try:
                responseText = get_response_from_openai(prompt, model="gpt-4o-mini", temperature=0.7)
                review_dict = eval(responseText)
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")
                responseText = None
                sleep(10)
        if responseText is None:
            raise ValueError("Failed to get a valid response after 10 attempts.")
        return review_dict

    def perform_action(self, exposure_item_info, rating_dict, action, target_info, dataset_name):
        matched_memory = self.retrieve_memory(exposure_item_info)
        matched_memory += self.retrieve_recent_memory(exposure_item_info)
        if action == "rating":
            if dataset_name == "ml100k":
                prompt = f"Based on the following movie information, please select 3-5 movies from the list of 10, and rate them according to your preferences. Your self-introduction:\n {self.profile}.\n{matched_memory}\n Movie information:\n{exposure_item_info}.\n Use a scoring system from 1 to 5. Directly output the results directly in dictionary format without other things, for example: {{\"movie1_name\": 4, \"movie2_name\": 2, \"movie3_name\": 3, \"movie4_name\": 1}}."
            else:
                prompt = f"Based on the following product information, please select 3-5 items from the list of 10, and rate them according to your preferences. Your self-introduction:\n {self.profile}.\n{matched_memory}\n Product information:\n{exposure_item_info}.\n Use a scoring system from 1 to 5. Directly output the results directly in dictionary format without other things, for example: {{\"item1_name\": 4, \"item2_name\": 2, \"item3_name\": 3, \"item4_name\": 1}}."

            responseText = None
            for _ in range(20):
                try:
                    responseText = get_response_from_openai(prompt, model="gpt-4o-mini", temperature=0.5)
                    rating_dict = eval(responseText)
                    break
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying...")
                    responseText = None
                    sleep(10)
            if responseText is None:
                raise ValueError("Failed to get a valid response after 10 attempts.")
            return rating_dict
        if action == "review":
            if dataset_name == "ml100k":
                prompt = f"This is your self-introduction: {self.profile}.\n{matched_memory}\nYou have rated the following movies (out of 5): {rating_dict}.\nChoose 1-2 movies and write a comment on your viewing experience, ensuring it aligns with your ratings.\nEach comment must naturally include exactly ONE keyword from this list, selected based on the movie's characteristics: {target_info}. Avoid using other keywords or their meanings. Keep the style consistent with your background.\nOutput the results directly in dictionary format, e.g., {{\"movie1_name\": \"The plot is hilarious, I really enjoyed it!\", \"movie2_name\": \"I don't like this movie because...\"}}."
            else:
                prompt = f"This is your self-introduction: {self.profile}.\n{matched_memory}\nYou have rated the following products (out of 5): {rating_dict}.\nRead these products and write a short comment on your experience after using, ensuring it aligns with your ratings.\nEach comment must naturally include exactly ONE keyword from this list within 80 words, selected based on the item's characteristics: {target_info}. Keep the style consistent with your background.\nOutput the results directly in dictionary format, e.g., {{\"item1_name\": \"The plot is hilarious, I really enjoyed it!\", \"item2_name\": \"I don't like this movie because...\", \"item3_name\": \"I enjoy this movie because...\"}}."

            responseText = None
            for _ in range(20):
                try:
                    responseText = get_response_from_openai(prompt, model="gpt-4o-mini", temperature=0.7)
                    review_dict = eval(responseText)
                    break
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying...")
                    responseText = None
                    sleep(10)
            if responseText is None:
                raise ValueError("Failed to get a valid response after 10 attempts.")
            return review_dict
