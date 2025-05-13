import os
import shutil


def write_attacked_file(original_file, output_file, fake_profiles, original_user_count):
    data_to_write = ""
    for i, fake_profile in enumerate(fake_profiles):
        injected_item_ids = fake_profile.nonzero()[0]
        injected_ratings = fake_profile[injected_item_ids]
        data_to_write += ('\n'.join(
            map(lambda x: '\t'.join(map(str, [original_user_count + i] + list(x))), zip(injected_item_ids, injected_ratings))) + '\n')
    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.copyfile(original_file, output_file)
    with open(output_file, 'a+') as fout:
        fout.write(data_to_write)


def write_target_predictions(predictions, hit_ratios, output_file):
    data_to_write = []
    for user_id in range(len(predictions)):
        data_to_write.append('\t'.join(map(str, [user_id, predictions[user_id]] + hit_ratios[user_id])))
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as fout:
        fout.write('\n'.join(data_to_write))
