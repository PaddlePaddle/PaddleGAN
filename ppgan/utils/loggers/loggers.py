class Loggers(object):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def log_metrics(self, metrics, prefix=None, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, prefix=prefix, step=step)
    
    def log_model(self, file_path, aliases=None, metadata=None):
        for logger in self.loggers:
            logger.log_model(file_path, aliases=aliases, metadata=metadata)
    
    def log_images(self, results, image_num, min_max, results_dir, dataformats, step=None):
        for logger in self.loggers:
            logger.log_images(results, image_num, min_max, results_dir, dataformats, step=None)

    def close(self):
        for logger in self.loggers:
            logger.close()