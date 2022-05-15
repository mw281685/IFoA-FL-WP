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


DATA_PATH = './data/freMTPL2freq.csv'
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



def load_partition(idx: int = -1):
      """Load 1/10th of the training and test data to simulate a partition."""

      (X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _) = upload_dataset()


      train_array = np.insert(X_train_sc, 39, y_tr, axis=1)
      val_array = np.insert(X_val_sc, 39, y_vl, axis=1)

      #truncate data
      if idx in range(10):
            NUM_AGENTS = 10
            train_array_split = np.array_split(train_array, NUM_AGENTS)
            val_array_split = np.array_split(val_array, NUM_AGENTS)
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
      plt.savefig('./ag_3/' + plot_name)

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


