class ApiResponse:
    def __init__(self, message="success", data=None):
        self.message = message
        self.data = data 
    def json(self):
        return {"message": self.message, "data": self.data}