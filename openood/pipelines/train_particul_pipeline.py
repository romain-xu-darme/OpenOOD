import openood.utils.comm as comm
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger


class TrainParticulPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)
            print('Start training...', flush=True)

        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_loss(net, val_loader, epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model if necessary and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # start calibration
            print('Start calibrating...', flush=True)
            net = recorder.load_model_state(net, 'best.ckpt')
            net = trainer.calibrate()
            recorder.save_model_state(net, 'best.ckpt')

        net = recorder.load_model_state(net, 'best.ckpt')
        train_metrics = evaluator.eval_conf(net, train_loader)
        test_metrics = evaluator.eval_conf(net, test_loader)
        if comm.is_main_process():
            print("Train metrics", train_metrics)
            print("Test metrics:", test_metrics)
            print('Completed!', flush=True)
