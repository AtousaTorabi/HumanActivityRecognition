from pylearn2.utils import serial
from pylearn2.train_extensions import TrainExtension

class MonitorBasedSaveLatest(TrainExtension):
    """
    A callback that saves a copy of the model every time it achieves
    a new minimal value of a monitoring channel.
    """
    def __init__(self, save_path):
        """
        Parameters
        ----------
        channel_name : str
            The name of the channel we want to minimize
        save_path : str
            Path to save the best model to
        """

        self.__dict__.update(locals())
        del self.self


    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            model.monitor must contain a channel with name given by \
            self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """

        monitor = model.monitor
        if (monitor.get_epochs_seen()%20==0 and monitor.get_epochs_seen()>100) :

           serial.save(self.save_path + "%05d" % monitor.get_epochs_seen() + '.pkl', model, on_overwrite = 'backup')
