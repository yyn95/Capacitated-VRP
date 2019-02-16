from cvrp_parser import CVRPFileParser

if __name__ == '__main__':

    parser = CVRPFileParser('instances/Vrp-Set-A/A-n32-k5.vrp')
    parser.parse()

    print(parser.data)
    print(parser.capacity)
    print(parser.distances)
    print(parser.demands)
