import torch as th
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = '../data/freMTPL2freq.csv'
SEED = 212



def seed_torch(seed=SEED):
      th.manual_seed(seed)
      random.seed(seed)
      th.cuda.manual_seed(seed)
      th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
      np.random.seed(seed)  # Numpy module.
      random.seed(seed)  # Python random module.
      th.manual_seed(seed)
      th.backends.cudnn.benchmark = True
      th.backends.cudnn.deterministic = True

#seed_torch() # for FL training, we cannot set seed, as everyone will have same rng

def upload_dataset():
      
      df = pd.read_csv(DATA_PATH)
#      df = df0.sort_values('ClaimNb', ignore_index=True)

      #transformations and corrections
      df['VehPower'] = df['VehPower'].astype(object) # categorical ordinal
      df['ClaimNb'].values[df['ClaimNb']>4] = 4 # corrected for unreasonable observations (see M.V. Wuthrich)
      df['VehAge'].values[df['VehAge']>20] = 20 # capped for NN training (see M.V. Wuthrich)
      df['DrivAge'].values[df['DrivAge']>90] = 90 # capped for NN training (see M.V. Wuthrich)
      df['BonusMalus'].values[df['BonusMalus']>150] = 150 # capped for NN training (see M.V. Wuthrich)
      df['Density']=np.log(df['Density']) # logged for NN training     (see M.V. Wuthrich)
      df['Exposure'].values[df['Exposure']>1] = 1 # corrected for unreasonable observations (see M.V. Wuthrich)
      df_new = df.drop(['IDpol'], axis=1) # variable not used

      #Encode the data as per Wuthrich
      df_new_encoded = pd.get_dummies(df_new, columns=['VehBrand', 'Region'], drop_first=True)

      cleanup_nums = {"Area":     {"A": 1, "B": 2, "C": 3, "D": 4, "E":5, "F": 6},
                  "VehGas":   {"Regular": 1, "Diesel": 2} }

      #Apply label encoding - NOT ONE-HOT/DUMMY
      df_new_encoded = df_new_encoded.replace(cleanup_nums)

      #Apply MinMaxScaler as per Wuthrich
      #TODO this should really be done for each train, val, test data
      
      scaler = MinMaxScaler()
      df_new_encoded[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']] = scaler.fit_transform(df_new_encoded[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']])


      #Convert to numpy array 
      df_array=df_new_encoded.to_numpy()

      # Create Input and Output Data
      X = df_array[:, 1:]
      y = df_array[:, 0]

      #Data Splitting
      #Split data into train and final test
      X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=SEED)
      
      #Split train into train and validation
      X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1,  random_state=SEED)

      train_array = np.insert(X_train, 39, y_train, axis=1)
      val_array = np.insert(X_val, 39, y_val, axis=1)


      return (X_train, X_val, X_test, y_train, y_val, y_test, df_new_encoded.columns.tolist()[1:], scaler)



def prep_partitions(agents:int = 10):
      (X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _) = upload_dataset()

      print(len(X_column_names))

      pd.DataFrame(X_test_sc[:,0:39], columns=X_column_names).to_csv('../data/X_test.csv', index=False)
      pd.DataFrame(y_te).to_csv('../data/y_test.csv', index=False)

      train_array = np.insert(X_train_sc, 39, y_tr, axis=1)
      val_array = np.insert(X_val_sc, 39, y_vl, axis=1)

      print(f'train_array shape = {train_array.shape}, val_array shape = {val_array.shape}')
  #    train_array = train_array[train_array[:, 39].argsort()] # sorting !!!

      #train_array_split = np.array_split(train_array, agents)

      val_array_split = np.array_split(val_array, agents)
      train_array_len = train_array.shape[0]
      idx_0 = 2*train_array_len//6
      idx_1 = idx_0 + 1*train_array_len//6
      print(f'idx_0 = {idx_0} , idx_1 = {idx_1}')
      #truncate datas
      for idx in range(agents):
            print(f'Processing idx = {idx}')
#            X_train_sc = train_array_split[idx][:,0:39]
#            pd.DataFrame(X_train_sc, columns=X_column_names).to_csv('./data/X_train_' + str(idx) + '.csv', index=False)
#            y_tr = train_array_split[idx][:, 39]
#            pd.DataFrame(y_tr).to_csv('./data/y_tr_' + str(idx) + '.csv', index=False)
            X_val_sc = val_array_split[idx][:, 0:39]
            pd.DataFrame(X_val_sc, columns=X_column_names).to_csv('../data/X_val_' + str(idx) + '.csv', index=False)
            y_vl = val_array_split[idx][:,39]
            pd.DataFrame(y_vl).to_csv('../data/y_vl_' + str(idx) + '.csv', index=False)

      X_train_sc = train_array[0:idx_0][:,0:39]
      pd.DataFrame(X_train_sc, columns=X_column_names).to_csv('../data/X_train_0.csv' , index=False)
      y_tr = train_array[0:idx_0][:, 39]
      pd.DataFrame(y_tr).to_csv('../data/y_tr_0.csv', index=False)

      X_train_sc = train_array[idx_0: idx_1][:,0:39]
      pd.DataFrame(X_train_sc, columns=X_column_names).to_csv('../data/X_train_1.csv' , index=False)
      y_tr = train_array[idx_0:idx_1][:, 39]
      pd.DataFrame(y_tr).to_csv('../data/y_tr_1.csv', index=False)

      X_train_sc = train_array[idx_1:][:,0:39]
      pd.DataFrame(X_train_sc, columns=X_column_names).to_csv('../data/X_train_2.csv' , index=False)
      y_tr = train_array[idx_1:][:, 39]
      pd.DataFrame(y_tr).to_csv('../data/y_tr_2.csv', index=False)


def load_partition(idx: int = -1, num_agents: int = 10):
      """Load 1/(num_agents) of the training and test data to simulate a partition."""

      (X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _) = upload_dataset()


      train_array = np.insert(X_train_sc, 39, y_tr, axis=1)
      val_array = np.insert(X_val_sc, 39, y_vl, axis=1)

      #truncate data
      if idx in range(num_agents):
            train_array_split = np.array_split(train_array, num_agents)
            val_array_split = np.array_split(val_array, num_agents)
            X_train_sc = train_array_split[idx][:,0:39]
            y_tr = train_array_split[idx][:, 39]
            X_val_sc = val_array_split[idx][:, 0:39]
            y_vl = val_array_split[idx][:,39]


      # Created tensordataset
      train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_sc).float(), torch.from_numpy(y_tr).float())
      val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_sc).float(), torch.from_numpy(y_vl).float())
      test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test_sc).float(), torch.from_numpy(y_te).float())
      
      return (train_dataset, val_dataset, test_dataset, X_column_names)


def load_individual_data(agent_id):
      MY_DATA_PATH = '../data'
      X_train_sc = pd.read_csv(MY_DATA_PATH + '/X_train_' + str(agent_id) + '.csv')
      X_column_names = X_train_sc.columns.tolist()

      y_tr = pd.read_csv(MY_DATA_PATH + '/y_tr_' + str(agent_id) +  '.csv')

      X_val_sc = pd.read_csv(MY_DATA_PATH + '/X_val_' + str(agent_id) + '.csv')
      y_vl = pd.read_csv(MY_DATA_PATH + '/y_vl_' + str(agent_id) + '.csv')

      X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
      y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')

      # Created tensordataset
      train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_sc.values).float(), torch.tensor(y_tr.values).float())
      val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val_sc.values).float(), torch.tensor(y_vl.values).float())
      test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test_sc.values).float(), torch.tensor(y_te.values).float())
      
      return (train_dataset, val_dataset, test_dataset, X_column_names, torch.tensor(X_test_sc.values).float())

#---------------------- Model predictions testing

def exp_model_predictions(model, test_loader):
      y_pred_list = []
      with th.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                  y_test_pred = model(X_batch)
                  y_pred_list.append(y_test_pred.cpu().numpy())
      y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
      return np.exp(y_pred_list)


def frequency_conversion(FACTOR, df, freq_dictionary):
      for key in freq_dictionary:
            df[freq_dictionary[key]]=df[key]/df['Exposure']

      df.insert(1,FACTOR+'_binned_midpoint',[round((a.left + a.right)/2,0) for a in df[FACTOR+'_binned']])


def one_way_graph(FACTOR, df, plot_name,  *freq):
      data_preproc = df[[FACTOR+'_binned_midpoint', *freq]]
      plt.figure(figsize=(15,8))
      sns.lineplot(data=pd.melt(data_preproc, [FACTOR+'_binned_midpoint']), x=FACTOR+'_binned_midpoint', y='value', hue='variable')
      #plt.show()
      plt.savefig('../ag_0/' + plot_name)

def predictions_check(run_name, model_global, model_partial, model_fl):
      
      (X_train, X_val, X_test, y_train, y_val, y_test, X_column_names, scaler) = upload_dataset()
      test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
      
      test_loader = DataLoader(dataset=test_dataset, batch_size=1)


      test_complete_data=np.column_stack((X_test, y_test))

      X_column_names.append('ClaimNb')

      #Convert dataset of test data, actuals, and prediction back into dataframe

      df_test=pd.DataFrame(data=test_complete_data,    # values
                     columns=X_column_names)  # 1st row as the column names

      df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']]=scaler.inverse_transform(df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']] )
      
      #Bin certain factors
      factor_list = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehGas', 'Density']
      BINSIZE = 15

      for i in factor_list:
          df_test[i+'_binned'] = pd.cut(df_test[i], bins=BINSIZE, duplicates='drop')


      y_pred_list_exp = exp_model_predictions(model_global, test_loader)
      df_test['ClaimNb_pred']=pd.Series(y_pred_list_exp)

      y_partial_pred_list_exp = exp_model_predictions(model_partial, test_loader)
      df_test['ClaimNb_partial_pred']=pd.Series(y_partial_pred_list_exp)
      
      y_fl_pred_list_exp = exp_model_predictions(model_fl, test_loader)
      df_test['ClaimNb_fl_pred']=pd.Series(y_fl_pred_list_exp)

      # One way analysis
      FACTOR = 'VehAge'

      df_sum=df_test.groupby([FACTOR+'_binned'])['Exposure','ClaimNb', 'ClaimNb_pred', 'ClaimNb_partial_pred', 'ClaimNb_fl_pred'].sum().reset_index()

      frequency_conversion(FACTOR, df_sum, {'ClaimNb':'freq', 'ClaimNb_pred':'freq_pred', 'ClaimNb_partial_pred':'freq_partial_pred', 'ClaimNb_fl_pred':'freq_fl_pred'})


      one_way_graph(FACTOR, df_sum, run_name, 'freq', 'freq_pred', 'freq_partial_pred','freq_fl_pred')


