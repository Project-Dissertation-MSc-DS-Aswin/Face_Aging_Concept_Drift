from whylogs import get_or_create_session

def setup_logger(logger_name):
  session = get_or_create_session()
  logger = session.logger(dataset_name=logger_name)

  return logger