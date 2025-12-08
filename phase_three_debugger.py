import datetime
class DebugLogger:
    def __init__(self, enabled=True, log_level='INFO'):
        self.enabled = enabled
        self.log_level = log_level
        self.levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        self.current_level = self.levels.get(log_level, 1)
        self.log_history = []
        
    def log(self, message, level='INFO', module=''):
        if not self.enabled:
            return
            
        if self.levels.get(level, 1) >= self.current_level:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] [{level:7s}] [{module:20s}] {message}"
            print(log_entry)
            self.log_history.append(log_entry)
            
            # Keep history bounded
            if len(self.log_history) > 1000:
                self.log_history.pop(0)
    
    def save_logs(self, filename='debug_log.txt'):
        with open(filename, 'w') as f:
            for entry in self.log_history:
                f.write(entry + '\n')

# Create global logger
logger = DebugLogger(enabled=True, log_level='INFO')