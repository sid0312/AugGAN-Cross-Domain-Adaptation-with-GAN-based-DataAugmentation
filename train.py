import time
from trainer import *
from network import *
import yaml
from yaml.loader import SafeLoader
from data.SYNTHIA_loader import ImageDataset
import argparse
#from data.GTA_loader import ImageDataset
#from data.GTA_loader_crop import ImageDataset
#from data.fake_real_loader import ImageDataset

torch.backends.cudnn.benchmark = True

def main():
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
    params = data['train']
    print('-----------------> preparing DataLoader')
    dataset_args = params['hyperparameters']['dataset']
    loader_args = params['hyperparameters']['loader']
    batch_size = params['hyperparameters']['network']['net']['batch_size']
    name = params['hyperparameters']['network']['net']['name']
    description = params['description']
    save_path = params['save_path']
    epochs = params['epochs']
    lr = params['hyperparameters']['network']['net']['lr']

    parser = argparse.ArgumentParser(description="Baseline Experiments")
    parser.add_argument("--epochs", type=int, default=epochs, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=batch_size, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=lr, help="Learning Rate.")
    parser.add_argument("--load-from-checkpoint", type=int, default=-1, help="Load from checkpoint if the value is not -1")
    parser.add_argument("--test-on-cpu", type=bool, default=False)

    parser_args = vars(parser.parse_args())
    params['hyperparameters']['network']['net']['lr'] = parser_args['lr']
    epochs = parser_args['epochs']
    test_on_cpu = parser_args['test_on_cpu']
    batch_size = parser_args['batch_size']

    train_loader = torch.utils.data.DataLoader(
        dataset=ImageDataset('../../input/synthia/SYNTHIA/', 'SYNTHIA-SEQS-{}-SPRING', 'SYNTHIA-SEQS-{}-NIGHT', 286, \
                            ['02'], 256, "train", 'scale_width'), batch_size=batch_size, **loader_args)
    val_loader = torch.utils.data.DataLoader(
        dataset=ImageDataset('../../input/synthia/SYNTHIA/', 'SYNTHIA-SEQS-{}-SPRING', 'SYNTHIA-SEQS-{}-NIGHT', 286, \
                            ['02'], 256, "val", 'scale_width'), batch_size=batch_size, **loader_args)
    test_loader = torch.utils.data.DataLoader(
        dataset=ImageDataset('../../input/synthia/SYNTHIA/', 'SYNTHIA-SEQS-{}-SPRING', 'SYNTHIA-SEQS-{}-NIGHT', 286, \
                            ['02'], 256, "test", 'scale_width'), batch_size=batch_size, **loader_args)

    print ("Train Size: ", len(train_loader))
    print ("Val Size: ", len(val_loader))
    print ("Test Size: ", len(test_loader))
    print('-----------------> preparing model: {}'.format(name))
    load_from_checkpoint = parser_args['load_from_checkpoint']
    net = network(params['hyperparameters']['network'])
    if load_from_checkpoint != -1:
        net.load_model_from_checkpoint("../output", "latest")
    coach = trainer(net, save_path, name, description)
    print('-----------------> start training')
    coach.train(train_data_loader=train_loader, val_data_loader=val_loader, epochs=epochs, test_data_loader=test_loader)
    print ('----------------> start testing')
    coach.evaluate(test_loader, test_on_cpu=test_on_cpu)
    # coach.visualize(test_loader)

if __name__ == '__main__':
    main()
