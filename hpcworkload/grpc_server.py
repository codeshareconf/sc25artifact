import grpc
from concurrent import futures
import proto.entity_pb2
import proto.entity_pb2_grpc

class OperatorServicer(proto.entity_pb2_grpc.OperatorServicer):
    def Operate(self, request, context):
        filename = request.filename
        print(f"Client requested file: {filename}")

        try:
            with open(filename, 'rb') as f:
                content = f.read()
            return proto.entity_pb2.Entity(entity=content)
        except FileNotFoundError:
            context.set_details(f'File {filename} not found!')
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return proto.entity_pb2.Entity()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    proto.entity_pb2_grpc.add_OperatorServicer_to_server(OperatorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()