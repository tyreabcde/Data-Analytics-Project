import data_processing
import os
save_path = r"C:\Users\USER\Desktop\Course\Data Analytics\Final Project"

train_source, train_target, test_source = data_processing.preprocess_data()

slot_train_pr = data_processing.time_slot_usage_time(train_source, value_type='proportion')
slot_test_pr = data_processing.time_slot_usage_time(test_source, value_type='proportion')

slot_train_log = data_processing.time_slot_usage_time(train_source, value_type='log')
slot_test_log = data_processing.time_slot_usage_time(test_source, value_type='log')

slot_train_pr.to_csv(os.path.join(save_path, 'slot_train_pr.csv'))
slot_test_pr.to_csv(os.path.join(save_path, 'slot_test_pr.csv'))
slot_train_log.to_csv(os.path.join(save_path, 'slot_train_log.csv'))
slot_test_log.to_csv(os.path.join(save_path, 'slot_test_log.csv'))