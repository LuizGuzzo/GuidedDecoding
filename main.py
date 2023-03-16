import os
import argparse

from training import Trainer
from evaluate import Evaluater

def get_args():
    file_dir = os.path.dirname(__file__) #Directory of this path

    parser = argparse.ArgumentParser(description='UpSampling for Monocular Depth Estimation')

    #Mode
    parser.set_defaults(train=True)
    parser.set_defaults(evaluate=True)
    parser.add_argument('--train',
                        dest='train',
                        action='store_true')
    parser.add_argument('--eval',
                        dest='evaluate',
                        action='store_true')
    
    parser.set_defaults(deep_supervision=False)
    parser.add_argument('--ds',
                        dest='deep_supervision',
                        action='store_true')
    

    #Data
    parser.add_argument('--data_path',
                        type=str,
                        help='path to train data',
                        default="./nyudata/CSVdata.zip")#os.path.join(file_dir, 'kitti_comb'))
    parser.add_argument('--test_path',
                        type=str,
                        help='path to test data',
                        default="./nyudata/CSVdata.zip") #os.path.join(file_dir, 'kitti_comb'))
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset for training',
                        choices=['kitti', 'nyu', 'nyu_reduced'],
                        default='nyu_reduced')
    parser.add_argument('--resolution',
                        type=str,
                        help='Resolution of the images for training',
                        choices=['full', 'half', 'mini', 'tu_small', 'tu_big'],
                        default="half")#'half')
    parser.add_argument('--eval_mode',
                        type=str,
                        help='Eval mode',
                        choices=['alhashim', 'tu'],
                        default='alhashim')


    #Model
    parser.add_argument('--model',
                        type=str,
                        help='name of the model to be trained',
                        default="teste")#'GuideDepth-S')
    parser.add_argument('--weights_path',
                        type=str,
                        help='path to model weights'
                        #,default="./results/best_model.pth" # FICA ESPERTO PARA LIGAR DE VOLTA QND FOR TESTAR
                        )

    #Checkpoint
    parser.add_argument('--load_checkpoint',
                        type=str,
                        help='path to checkpoint',
                        default=""#./checkpoints/checkpoint_19.pth" # FICA ESPERTO PARA LIGAR DE VOLTA QND FOR TESTAR
                        )
    parser.add_argument('--save_checkpoint',
                        type=str,
                        help='path to save checkpoints to',
                        default='./checkpoints')
    parser.add_argument('--save_results',
                        type=str,
                        help='path to save results to',
                        default='./results')

    #Optimization
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size',
                        default=2)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate',
                        default=1e-4)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of epochs',
                        default=20)
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        help='step size of the scheduler',
                        default=15)

    #System
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of dataloader workers',
                        default=0)


    return parser.parse_args()


def main():
    args = get_args()
    # print(args)

    print("deep_supervision activated? : ",args.deep_supervision)

    if args.train:
        model_trainer = Trainer(args)
        model_trainer.train()
        args.weights_path = os.path.join(args.save_results, 'best_model.pth')

    if args.evaluate:
        evaluation_module = Evaluater(args)
        evaluation_module.evaluate()

    print("batch_size:",args.batch_size)

if __name__ == '__main__':
    main()


"""
python main.py --train --eval --data_path "./nyudata/CSVdata.zip" --test_path "./nyudata/CSVdata.zip" --dataset "nyu_reduced" --resolution "full" --eval_mode "alhashim" --model "teste" --batch_size 20 --num_workers 0
"""