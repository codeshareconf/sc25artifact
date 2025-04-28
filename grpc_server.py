import grpc
from concurrent import futures
import proto.entity_pb2
import proto.entity_pb2_grpc

class OperatorServicer(proto.entity_pb2_grpc.OperatorServicer):
    def Operate(self, request, context):
        result = request.entity
        return proto.entity_pb2.Response(response = "success")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    proto.entity_pb2_grpc.add_OperatorServicer_to_server(OperatorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()