import argparse
from algorithms import ant_is
from algorithms import ant_is_multithread

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="run the ant colony algorithm on the given csv")
    parser.add_argument("column", help="which column needs to be searched")
    parser.add_argument("-j", "--jobs", type=int, help="number of threads to execute (default=1)")
    args = parser.parse_args() 
    
    print(args)
    
    # running the sequential version
    if not args.jobs:
        ant_is.ant_colony(args.file, args.column)

    # running the parallel version
    else:
        pass

if __name__ == '__main__':
    main()