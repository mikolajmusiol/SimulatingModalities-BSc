from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, log_dir):
        self.train_writer = SummaryWriter(log_dir=log_dir + '\\training')
        self.validation_writer = SummaryWriter(log_dir=log_dir+'\\validation')

    def log_train(self, gan_loss, disc_loss, metrics, step):
        self.__log(gan_loss, disc_loss, metrics, step, self.train_writer)

    def log_validation(self, gan_loss, disc_loss, metrics, step):
        self.__log(gan_loss, disc_loss, metrics, step, self.validation_writer)

    def __log(self, gan_loss, disc_loss, metrics, step, writer):
        writer.add_scalar("Loss:Generator", gan_loss, step)
        writer.add_scalar("Loss:Discriminator", disc_loss, step)
        writer.add_scalar("Metrics:MSE", metrics.mse, step)
        writer.add_scalar("Metrics:NMSE", metrics.nmse, step)
        writer.add_scalar("Metrics:RMSE", metrics.rmse, step)
        writer.add_scalar("Metrics:MAE", metrics.mae, step)
        writer.add_scalar("Metrics:SNR", metrics.snr, step)
        writer.add_scalar("Metrics:PSNR", metrics.psnr, step)
        writer.add_scalar("Metrics:SSIM", metrics.ssim, step)
        writer.close()