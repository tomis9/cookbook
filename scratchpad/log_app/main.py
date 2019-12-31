import logging
import sys
from datetime import datetime
# import logstash
from pythonjsonlogger import jsonlogger


# logging configuration ----
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record,
                                                    message_dict)
        if not log_record.get('timestamp'):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
hdl_file = logging.FileHandler('./logs.log')
hdl_file.setFormatter(logging.Formatter(format))
hdl_json_stdout = logging.StreamHandler(sys.stdout)
formatter = CustomJsonFormatter('(timestamp) (level) (name) (message)')
hdl_json_stdout.setFormatter(formatter)

logger.addHandler(hdl_file)
logger.addHandler(hdl_json_stdout)
# ----

logger.debug('some debug')
