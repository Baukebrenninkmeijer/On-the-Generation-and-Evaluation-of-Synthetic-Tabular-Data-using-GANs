from comet_ml import Experiment
import pandas as pd
import argparse
import os
from tgan_wgan_gp.model import TGANModel
## Change above line to which version you want to use. Choose from ['tgan_org', tgan_skip', 'tgan_wgan_gp']

def get_data(ds, drop=None, n_unique=20, sep=';', suffix='cat'):
    d = pd.read_csv(f'../data/{ds}/{ds}_{suffix}.csv', sep=sep)
    if drop is not None:
        d = d.drop(drop, axis=1)

    continuous_columns = []
    for col in d._get_numeric_data().columns:
        if len(d[col].unique()) > n_unique:
            continuous_columns.append(d.columns.get_loc(col))
    return d, continuous_columns

parser = argparse.ArgumentParser(description='Evaluate data synthesizers')
parser.add_argument('--dataset', nargs='*', help='Which dataset to choose. Options are berka, creditcard and ticket', default=['berka', 'census', 'creditcard'])

args = parser.parse_args()
datasets = args.dataset

for ds in datasets:

    if ds == 'berka':
        d, continuous_columns = get_data(ds, drop=['trans_bank_partner', 'trans_account_partner'])
    elif ds == 'census':
        d, continuous_columns = get_data(ds, sep=',')
    elif ds == 'creditcard':
        d, continuous_columns = get_data(ds, sep=',', suffix='num')
    else:
        raise Exception('Unknown dataset mentioned')

    project_name = "tgan-wgan-gp"
    experiment = Experiment(api_key=os.environ['COMETML_API_KEY'],
                            project_name=project_name, workspace="baukebrenninkmeijer")
    experiment.log_parameter('dataset', ds)
    print(f'ds: {ds}')

    batch_size = 200
    assert len(d) > batch_size, f'Batch size larger than data'
    steps_per_epoch = len(d)//batch_size
    print('Steps per epoch: ', steps_per_epoch)
    tgan = TGANModel(continuous_columns,
                     restore_session=False,
                     max_epoch=100,
                     steps_per_epoch=steps_per_epoch,
                     batch_size=batch_size,
                     experiment=experiment,
                     num_gen_rnn=50,
                     num_gen_feature=64)
    tgan.fit(d)

    try:
        if os.path.exists('/mnt'):
            if not os.path.exists('/mnt/model'):
                os.mkdir('/mnt/model')
            model_path = f'/mnt/model/{ds}_{project_name}'
        else:
            model_path = f'model/{ds}_{project_name}'
    except:
        model_path = f'model/{ds}_{project_name}'

    # try:
    #     tgan.save(model_path, force=True)
    # except Exception as e:
    #     print(f'{e}\nModel could not be saved')
    #
    num_samples = 100000
    new_samples = tgan.sample(num_samples)
    new_samples.to_csv(f'temp_save_{ds}.csv', index=False)

    p = new_samples.copy()
    d.columns = p.columns
    if ds == 'berka' or ds == 'census':
        p[p._get_numeric_data().columns] = p[p._get_numeric_data().columns].astype('int')
    if ds == 'creditcard':
        p[['Time', 'Class']] = p[['Time', 'Class']].astype('int')

    try:
        if os.path.exists('/mnt'):
            if not os.path.exists('/mnt/samples'):
                os.mkdir('/mnt/samples')
            p.to_csv(f'/mnt/samples/{ds}_sample_{project_name}.csv', index=False)
        else:
            p.to_csv(f'samples/{ds}_sample_{project_name}.csv', index=False)
    except:
        p.to_csv(f'samples/{ds}_sample_{project_name}.csv', index=False)

    try:
        os.remove(f'temp_save_{ds}.csv')
    except Exception as e:
        print(f'{e} -- Could not remove temp_save_{ds}.csv')

    experiment.end()


    import tensorflow as tf
    tf.keras.backend.clear_session()

