import torch as th
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from run_config import NUM_FEATURES, DATA_PATH, SEED


#---------------------- DATASET CHECKS: ----------------------------------------------------------------------------

import pandas as pd

def row_check(agents: int = 10):
    """
    Validates the integrity of the dataset across multiple agents by checking the total number of rows, 
    the sum of exposure values, and the sum of claims across training, validation, and test datasets.

    This function ensures that the combined dataset from multiple agents, along with the test dataset,
    matches expected values for total row count, total exposure, and total claims. These checks are critical for
    verifying data integrity and consistency before proceeding with further data analysis or model training.

    Parameters:
        agents : int, optional
            The number of agents (or partitions) for which training and validation datasets are available.
            Defaults to 10.

    Raises:
        AssertionError
            If the total number of rows, total exposure, or total claims do not match expected values, 
            an AssertionError is raised indicating which specific integrity check has failed.

    Notes:
        - Assumes existence of CSV files in '../data/' following specific naming conventions.
        - Useful for data preprocessing in machine learning workflows involving multiple sources or agents.
        - 'Exposure' and '0' are assumed to be column names in the respective CSV files for exposure and claims.

    Example:
        >>> row_check(agents=5)
        # Checks datasets for 5 agents, along with the test dataset, and prints the status of each check.
    """
    # Helper function to load a CSV file into a DataFrame
    def load_data_frame(prefix: str, index: int) -> pd.DataFrame:
        return pd.read_csv(f'../data/{prefix}_{index}.csv')

    # Initialize expected values
    expected_row_count = 678_013
    expected_exposure_sum = 358_360.11
    expected_claims_sum = 36_056

    # Load and aggregate datasets
    datasets = {'X_train': [], 'X_val': [], 'y_train': [], 'y_val': []}
    for prefix in datasets.keys():
        for i in range(agents):
            datasets[prefix].append(load_data_frame(prefix, i))
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')

    # Calculate totals
    total_row_count = sum(df.shape[0] for prefix in ['X_train', 'X_val'] for df in datasets[prefix]) + X_test.shape[0]
    total_exposure_sum = sum(df['Exposure'].sum() for prefix in ['X_train', 'X_val'] for df in datasets[prefix]) + X_test['Exposure'].sum()
    total_claims_sum = sum(df['0'].sum() for prefix in ['y_train', 'y_val'] for df in datasets[prefix]) + y_test['0'].sum()

    # Validate dataset integrity
    assert total_row_count == expected_row_count, f"Total row count mismatch: expected {expected_row_count}, got {total_row_count}"
    assert round(total_exposure_sum, 2) == round(expected_exposure_sum, 2), f"Total exposure mismatch: expected {expected_exposure_sum}, got {round(total_exposure_sum, 2)}"
    assert total_claims_sum == expected_claims_sum, f"Total claims sum mismatch: expected {expected_claims_sum}, got {total_claims_sum}"

    print('All checks passed successfully.')



#---------------------------- SKORCH_TUNING_UTILS ------------------------------------------------------
def training_loss_curve(estimator, agent_id):
    """
    Plots the training and validation loss curves along with the percentage of Poisson Deviance Explained (PDE).

    Parameters:
        estimator : object
            The trained model or estimator that contains the training history. It is expected
            to have a 'history' attribute that is a NumPy array or a similar structure with
            'train_loss', 'valid_loss', and 'weighted_PDE_best' columns.
        agent_id : int or str
            Identifier for the agent. Used for titling the plot and naming the saved figure file.
    
    Notes:
        - This function saves the generated plot as a PNG file in a directory named after the agent.
        - Ensure the directory '../ag_{agent_id}/' exists or adjust the save path accordingly.
        - The function uses matplotlib for plotting and requires this library to be installed.
    """
    
    # Creating a DataFrame from the estimator's history for easier manipulation
    train_val_loss_df = pd.DataFrame(estimator.history, columns=['train_loss', 'valid_loss', 'weighted_PDE_best'])
    train_val_loss_df.rename(columns={'weighted_PDE_best': 'PDE'}, inplace=True)

    # Plotting setup
    fig, ax1 = plt.subplots(figsize=(40, 15))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(train_val_loss_df['train_loss'], label='Training Loss', color='tab:blue')
    ax1.plot(train_val_loss_df['valid_loss'], label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y')
    ax1.grid()
    ax1.legend(loc='upper left')

    # Twin axis for PDE
    ax2 = ax1.twinx()
    ax2.set_ylabel('% of Poisson Deviance Explained', color='g')
    ax2.plot(train_val_loss_df['PDE'], label='PDE', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.legend(loc='upper right')

    # Title and save figure
    plt.title(f"Agent {agent_id}'s Best Model's Training Loss Curve")
    plt.savefig(f'../ag_{agent_id}/agent_{agent_id}_training_loss_chart.png', facecolor='white')
    plt.close(fig)



def hyperparameter_counts(dataframe, hyperparameter, x_label, title, name):
    """
    Plots and saves a bar chart of the value counts for a specified hyperparameter in a given DataFrame.

    This function visualizes the distribution of values for a selected hyperparameter within a dataset, 
    highlighting the frequency of each unique value. The resulting bar chart is saved to a file.

    Parameters:
        dataframe : pandas.DataFrame
            The DataFrame containing the data from which to count the hyperparameter values.
        hyperparameter : str
            The name of the column in `dataframe` representing the hyperparameter whose distribution is to be plotted.
        x_label : str
            The label for the x-axis of the plot, typically the name of the hyperparameter.     
        title : str
            The title of the plot, describing what the plot shows.    
        name : str
            The filename under which the plot will be saved. The plot is saved in the '../results/' directory.
            The '.png' extension is recommended to be included in the `name` for clarity.

    Examples:
        >>> df = pd.DataFrame({'model_depth': [2, 3, 4, 2, 3, 3, 2]})
        >>> hyperparameter_counts(df, 'model_depth', 'Model Depth', 'Distribution of Model Depths', 'model_depth_distribution.png')
        # This will create and save a bar chart visualizing the frequency of different model depths in the dataset.

    Notes:
        - The plot is saved with a white background to ensure readability when viewed on various devices.
        - Ensure the '../results/' directory exists before calling this function, or adjust the save path accordingly.
        - The function does not return any value. It directly saves the generated plot to a file.
    """
      
    fig, ax = plt.subplots(figsize=(10, 8))
    dataframe[hyperparameter].value_counts().plot(kind='bar', ax=ax)
    plt.grid()
    plt.xlabel(x_label)
    plt.xticks(rotation=0)
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig('../results/'+name, facecolor='white')



def load_individual_skorch_data(agent_id):
    """
    Loads training, validation, and test datasets for a specified agent or for global model training.

    This function reads the datasets from CSV files. If `agent_id` is -1, it loads the global datasets.
    Otherwise, it loads the agent-specific datasets based on the provided `agent_id`.

    Parameters:
        agent_id : int
            The identifier for the specific agent's dataset to load. If set to -1, the function loads the
            global training, validation, and test datasets.

    Returns:
        tuple
            A tuple containing the training features (X_train_sc), training labels (y_tr), validation features
            (X_val_sc), validation labels (y_vl), test features (X_test_sc), test labels (y_te), column names
            of the training features (X_column_names), and the total exposure from the training set (exposure).

    Examples:
        >>> X_train, y_train, X_val, y_val, X_test, y_test, column_names, exposure = load_individual_skorch_data(-1)
        >>> print(f"Training data shape: {X_train.shape}")
    """
    MY_DATA_PATH = '../data'
    suffix = '' if agent_id == -1 else f'_{agent_id}'
    
    X_train_sc = pd.read_csv(f'{MY_DATA_PATH}/X_train{suffix}.csv')
    X_column_names = X_train_sc.columns.tolist()

    y_tr = pd.read_csv(f'{MY_DATA_PATH}/y_tr{suffix}.csv')
    X_val_sc = pd.read_csv(f'{MY_DATA_PATH}/X_val{suffix}.csv')
    y_vl = pd.read_csv(f'{MY_DATA_PATH}/y_vl{suffix}.csv')
    
    # Note: It assumes X_test and y_test are the same across all agents
    X_test_sc = pd.read_csv(f'{MY_DATA_PATH}/X_test.csv')
    y_te = pd.read_csv(f'{MY_DATA_PATH}/y_test.csv')

    exposure = sum(X_train_sc['Exposure'])
    
    return (X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te, X_column_names, exposure)


#-------------------- FL training utils: ---------------------------
def seed_torch(seed=SEED):
    """
    Seeds the random number generators of PyTorch, NumPy, and Python's `random` module to ensure
    reproducibility of results across runs when using PyTorch for deep learning experiments.

    This function sets the seed for PyTorch (both CPU and CUDA), NumPy, and the Python `random` module,
    enabling CuDNN benchmarking and deterministic algorithms. It is crucial for experiments requiring
    reproducibility, like model performance comparisons. Note that enabling CuDNN benchmarking and
    deterministic operations may impact performance and limit certain optimizations.

    Parameters:
        seed : int, optional
            The seed value to use for all random number generators. The default value is `SEED`, which
            should be defined beforehand.

    Returns:
        None
            This function does not return a value but sets the random seed for various libraries.

    Notes:
        - When using multiple GPUs, `th.cuda.manual_seed_all(seed)` ensures all GPUs are seeded, 
        crucial for reproducibility in multi-GPU setups.

    Example:
        >>> SEED = 42
        >>> seed_torch(SEED)
    """
    th.manual_seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = True


#----------------------- Data preparation utils: --------------------------------
    
def preprocess_dataframe(df):
    """
    Applies preprocessing steps to the dataframe, including shuffling, data type transformations,
    and value capping based on specified criteria.

    Parameters:
        df : DataFrame 
            The pandas DataFrame to preprocess.

    Returns:
        DataFrame
            The preprocessed DataFrame.

    Usage:
    ```
    df_preprocessed = preprocess_dataframe(df)
    ```
    """    
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Data type transformations and capping
    df['VehPower'] = df['VehPower'].astype(object)  # Convert to categorical
    df['ClaimNb'] = pd.to_numeric(df['ClaimNb'], errors='coerce').clip(upper=4)
    df['VehAge'] = df['VehAge'].clip(upper=20)
    df['DrivAge'] = df['DrivAge'].clip(upper=90)
    df['BonusMalus'] = df['BonusMalus'].clip(upper=150)
    df['Density'] = np.log(df['Density'])
    df['Exposure'] = df['Exposure'].clip(upper=1)
    
    # Drop unused variable
    df = df.drop(['IDpol'], axis=1)
    
    return df

def encode_and_scale_dataframe(df):
    """
    Encodes categorical variables and scales numerical features within the DataFrame.

    Parameters:
        df : DataFrame 
            The DataFrame to encode and scale.

    Returns:
        DataFrame 
            The encoded and scaled DataFrame.
        MinMaxScaler 
            The scaler used for numerical feature scaling.

    Usage:
    ```
    df_encoded, scaler = encode_and_scale_dataframe(df_preprocessed)
    ```
    """
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['VehBrand', 'Region'], drop_first=True)
    
    # Label encoding
    cleanup_nums = {"Area": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6},
                    "VehGas": {"Regular": 1, "Diesel": 2}}
    df_encoded.replace(cleanup_nums, inplace=True)
    
    # Scale features
    scaler = MinMaxScaler()
    features_to_scale = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])
    
    return df_encoded, scaler

def split_data(df_encoded):
    """
    Splits the encoded DataFrame into training, validation, and test sets.

    Parameters:
        df_encoded : DataFrame 
            The encoded DataFrame from which to split the data.

    Returns:
        tuple
            Contains training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).

    Usage:
    ```
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)
    ```
    """
    X = df_encoded.iloc[:, 1:].to_numpy()
    y = df_encoded.iloc[:, 0].to_numpy()
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=SEED)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def upload_dataset():
    """
    Uploads, preprocesses, encodes, scales, and splits the dataset into training, validation, and test sets.

    Assumes the existence of a global `DATA_PATH` variable pointing to the dataset's location and a `SEED` for reproducibility.

    Returns:
        tuple
            Contains the training, validation, and test sets, feature names, and the scaler.

    Usage:
    ```
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = upload_dataset()
    ```
    """
    seed_torch()
    
    df = pd.read_csv(DATA_PATH)
    df_preprocessed = preprocess_dataframe(df)
    df_encoded, scaler = encode_and_scale_dataframe(df_preprocessed)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)
    
    feature_names = df_encoded.columns.tolist()[1:]
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler)


def load_individual_data(agent_id, include_val_in_train=False):
    """
    Loads individual or global datasets as PyTorch TensorDatasets, with an option to include validation data in the training set.

    This function dynamically loads training, validation, and test data from CSV files located in a specified directory.
    It can load data for a specific agent by ID or global data if the agent ID is set to -1. There is an option to merge
    training and validation datasets for scenarios where validation data should be included in training, e.g., for certain
    types of model tuning.

    Parameters:
        agent_id : int
            The identifier for the agent's dataset to load. If set to -1, global datasets are loaded.
        include_val_in_train : bool, optional
            Determines whether validation data is included in the training dataset. Default is False.

    Returns:
        tuple
            A tuple containing the training dataset, validation dataset, test dataset, column names of the training features,
            a tensor of test features, and the total exposure calculated from the training (and optionally validation) dataset.

    Examples:
        >>> train_dataset, val_dataset, test_dataset, column_names, test_features, exposure = load_individual_data(-1, True)
        >>> print(f"Training dataset size: {len(train_dataset)}")
    """
    MY_DATA_PATH = '../data'
    suffix = '' if agent_id == -1 else f'_{agent_id}'
    
    # Load datasets
    X_train = pd.read_csv(f'{MY_DATA_PATH}/X_train{suffix}.csv')
    y_train = pd.read_csv(f'{MY_DATA_PATH}/y_tr{suffix}.csv')
    X_val = pd.read_csv(f'{MY_DATA_PATH}/X_val{suffix}.csv')
    y_val = pd.read_csv(f'{MY_DATA_PATH}/y_vl{suffix}.csv')
    X_test = pd.read_csv(f'{MY_DATA_PATH}/X_test.csv')  # Assuming test data is the same for all agents
    y_test = pd.read_csv(f'{MY_DATA_PATH}/y_test.csv')

    # Merge training and validation datasets if specified
    if include_val_in_train:
        X_train = pd.concat([X_train, X_val], ignore_index=True)
        y_train = pd.concat([y_train, y_val], ignore_index=True)

    # Calculate exposure
    exposure = X_train['Exposure'].sum()

    # Convert to TensorDatasets
    train_dataset = TensorDataset(th.tensor(X_train.values).float(), th.tensor(y_train.values).float())
    val_dataset = TensorDataset(th.tensor(X_val.values).float(), th.tensor(y_val.values).float())
    test_dataset = TensorDataset(th.tensor(X_test.values).float(), th.tensor(y_test.values).float())

    return (train_dataset, val_dataset, test_dataset, X_train.columns.tolist(), th.tensor(X_test.values).float(), exposure)



def uniform_partitions(agents: int = 10, num_features: int = None):
    """
    Splits and saves the dataset into uniform partitions for a specified number of agents.

    This function loads a dataset via a previously defined `upload_dataset` function, then partitions
    the training and validation datasets uniformly across the specified number of agents. Each partition
    is saved to CSV files, containing both features and labels for each agent's training and validation datasets.

    Parameters:
        agents : int, optional
            The number of agents to split the dataset into. Defaults to 10.
        num_features : int, optional
            The number of features in the dataset. Automatically inferred if not specified.

    Notes:
        - Requires `upload_dataset` and `seed_torch` to be defined and accessible within the scope.
        - Saves partitioned data files in the '../data/' directory.

    Example:
        >>> uniform_partitions(agents=5)
        Creates and saves 5 sets of training and validation data for 5 agents, storing them in '../data/'.

    Raises:
        FileNotFoundError
            If the '../data/' directory does not exist or cannot be accessed.
    
    Returns:
        None
            The function does not return a value but saves partitioned datasets to disk.
    """
    # Load the dataset
    X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _ = upload_dataset()
    num_features = num_features or X_train_sc.shape[1]
    
    # Define the base path for saving files
    base_path = '../data/'
    
    # Function to save datasets to CSV
    def save_to_csv(data, filename, column_names=X_column_names):
        pd.DataFrame(data, columns=column_names).to_csv(f'{base_path}{filename}', index=False)
    
    # Save the global datasets
    save_to_csv(X_train_sc, 'X_train.csv')
    save_to_csv(y_tr, 'y_tr.csv', ['y'])
    save_to_csv(X_val_sc, 'X_val.csv')
    save_to_csv(y_vl, 'y_vl.csv', ['y'])
    save_to_csv(X_test_sc[:, :num_features], 'X_test.csv')
    save_to_csv(y_te, 'y_test.csv', ['y'])

    # Prepare and shuffle data
    seed_torch()
    train_data = np.hstack((X_train_sc, y_tr.reshape(-1, 1)))
    val_data = np.hstack((X_val_sc, y_vl.reshape(-1, 1)))
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    # Split and save partitioned data
    for i in range(agents):
        partition_train = np.array_split(train_data, agents)[i]
        partition_val = np.array_split(val_data, agents)[i]

        save_to_csv(partition_train[:, :num_features], f'X_train_{i}.csv')
        save_to_csv(partition_train[:, num_features:], f'y_tr_{i}.csv', ['y'])
        save_to_csv(partition_val[:, :num_features], f'X_val_{i}.csv')
        save_to_csv(partition_val[:, num_features:], f'y_vl_{i}.csv', ['y'])


#-------------------- results analysis utils: -------------------------------------------
def lorenz_curve(y_true, y_pred, exposure):
    """
    Calculates the Lorenz curve for given true values, predicted values, and exposures.

    The Lorenz curve is a graphical representation of the distribution of income or wealth. In this context,
    it is used to show the distribution of claims or losses in insurance, ordered by predicted risk. This function
    calculates the cumulative percentage of claims and exposures, sorted by the predicted risk.

    Parameters:
        y_true : array_like
            The true values of the claims or losses.
        y_pred : array_like
            The predicted risk scores associated with each claim or loss.
        exposure : array_like
            The exposure values associated with each observation.

    Returns:
        tuple of numpy.ndarray
            A tuple containing two arrays: the cumulative percentage of exposure and the cumulative percentage of claims,
            both sorted by the predicted risk.

    Examples:
        >>> y_true = np.array([100, 50, 20])
        >>> y_pred = np.array([0.2, 0.5, 0.1])
        >>> exposure = np.array([1, 2, 1])
        >>> cumulated_exposure, cumulated_claims = lorenz_curve(y_true, y_pred, exposure)
        >>> print(cumulated_exposure)
        >>> print(cumulated_claims)
    """
    # Convert inputs to numpy arrays and reshape for consistency
    y_true, y_pred, exposure = map(lambda x: np.asarray(x).flatten(), (y_true, y_pred, exposure))

    # Order samples by increasing predicted risk
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]

    # Calculate cumulative claims and exposure, then normalize
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure) / np.sum(ranked_frequencies * ranked_exposure)
    cumulated_exposure = np.cumsum(ranked_exposure) / np.sum(ranked_exposure)
    
    return cumulated_exposure, cumulated_claims




