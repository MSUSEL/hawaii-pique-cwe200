import json

class ProgressTracker:
    def __init__(self, total_steps, progress_type):
        self.total_steps = total_steps
        self.current_progress = 0
        self.progress_type = progress_type
        self.last_progress_percentage = -1  # Keep track of the last printed progress percentage

    def update_progress(self, step_increment):
        if self.total_steps > 0:
            self.current_progress += step_increment
            progress_percentage = min(round((self.current_progress / self.total_steps) * 100), 100)
            if progress_percentage != self.last_progress_percentage:  # Only print if the percentage has changed
                print(json.dumps({'type': self.progress_type, 'progress': progress_percentage}), flush=True)
                self.last_progress_percentage = progress_percentage  # Update the last progress percentage
        else:
            print(json.dumps({'type': self.progress_type, 'progress': 100}))

    def complete(self):
        self.current_progress = self.total_steps
        print(json.dumps({'type': self.progress_type, 'progress': 100}))