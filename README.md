# logStoreOpt: Logistic Storage Optimization

## Description

Python-based prototype for the optimization of the incoming orders to a logistic center. 

- Store is divided in different storage **locations**.
- Each location has a **fixed surface** and **access time**, and **holds stacks of different items**. 
- Only items of the **same product** may be **stacked together**, at the price of **increased access time per level**.
- Each **product** has a defined **surface** and a **stack-height limit**.
- Locations may or may not be **shelves**. Some products cannot be stored in shelves.
- The loction of the incoming **orders** (groups of **items** of different **products**) are optimized so as to **minimise used surface and access time**.

## Algorithm

The solutions of the the problem is found through a local search based algortihm, where several randomly generated initial solutions are optimized in parallel. The cost functions are normalized based on randomly generated solutions and scalarized using a weighted geometric mean. One may set the weights to prioritize one objective over another.

## Environment / Setup

```bash
git clone https://github.com/rcote98/logStoreOpt.git    # clone the repo
cd logStoreOpt                                          # move in the folder
python3 -m venv logStoreOpt_env                         # create virtualenv
source logStoreOpt_env/bin/activate                     # activate it
pip install -r requirements.txt                         # install dependencies
python3 example.py                                      # run the example code
```
