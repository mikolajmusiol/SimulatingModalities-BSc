from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.train_writer = SummaryWriter(log_dir=log_dir + '/training')
        self.validation_writer = SummaryWriter(log_dir=log_dir + '/validation')

    def log_train(self, gen_loss, disc_loss, metrics, step):
        self.__log(gen_loss, disc_loss, metrics, step, self.train_writer)

    def log_validation(self, gen_loss, disc_loss, metrics, step):
        self.__log(gen_loss, disc_loss, metrics, step, self.validation_writer)

    def __log(self, gen_loss, disc_loss, metrics, step, writer):
        writer.add_scalar("Loss/Generator", gen_loss, step)
        writer.add_scalar("Loss/Discriminator", disc_loss, step)

        for metric_name, value in metrics.results.items():
            writer.add_scalar(f"Metrics/{metric_name.upper()}", value, step)
