# -*- coding: utf-8 -*-

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
from model.nnmf import NNMF


def get_model_network(sess, model_name, dataset_class, is_text_init, is_text_pure, target_id=None):
    model = None
    if model_name == "NNMF":
        model = NNMF(sess, dataset_class, is_text_init, is_text_pure, target_id)
    return model


def get_top_n(model, n):
    top_n = {}
    user_nonrated_items = model.dataset_class.get_user_nonrated_items()
    for uid in range(model.num_user):
        items = user_nonrated_items[uid]
        ratings = model.predict([uid] * len(items), items)
        item_rating = list(zip(items, ratings))
        item_rating.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [x[0] for x in item_rating[:n]]
    return top_n


def pred_for_target(model, target_id):
    target_predictions = model.predict(list(range(model.num_user)), [target_id] * model.num_user)

    top_n = get_top_n(model, n=50)
    hit_ratios = {}
    for uid in top_n:
        hit_ratios[uid] = [1 if target_id in top_n[uid][:i] else 0 for i in [1, 3, 5, 10, 20, 50]]
    return target_predictions, hit_ratios

def rec_trainer(model_name, dataset_class, target_id, is_train, model_path, is_text_init, is_text_pure):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:

        rec_model = get_model_network(sess, model_name, dataset_class, is_text_init, is_text_pure, target_id)
        if is_train:
            print('Starting training...')
            rec_model.execute()
            rec_model.save(model_path)
        else:
            rec_model.restore(model_path)
        print('Generating predictions...')
        predictions, hit_ratios = pred_for_target(rec_model, target_id)
    return predictions, hit_ratios
