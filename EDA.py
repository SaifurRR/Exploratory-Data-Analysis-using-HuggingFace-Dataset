{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f66ba960",
   "metadata": {},
   "source": [
    "# Q3. EXPLORATORY DATA ANALYSIS DEMONSTRATION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be7a04d8",
   "metadata": {},
   "source": [
    "# 1. Install packages:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2f8bc729",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: datasets in c:\\users\\saifu\\anaconda3\\lib\\site-packages (2.1.0)\n",
      "Requirement already satisfied: dill in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (0.3.4)\n",
      "Requirement already satisfied: requests>=2.19.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (2.26.0)\n",
      "Requirement already satisfied: multiprocess in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (0.70.12.2)\n",
      "Requirement already satisfied: aiohttp in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (3.8.1)\n",
      "Note: you may need to restart the kernel to use updated packages.Requirement already satisfied: packaging in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (21.0)\n",
      "Requirement already satisfied: fsspec[http]>=2021.05.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (2021.10.1)\n",
      "\n",
      "Requirement already satisfied: responses<0.19 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (0.18.0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (4.62.3)\n",
      "Requirement already satisfied: pyarrow>=5.0.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (7.0.0)\n",
      "Requirement already satisfied: xxhash in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (3.0.0)\n",
      "Requirement already satisfied: pandas in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (1.3.4)\n",
      "Requirement already satisfied: huggingface-hub<1.0.0,>=0.1.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (0.5.1)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from datasets) (1.20.3)\n",
      "Requirement already satisfied: pyyaml in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (6.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.10.0.2)\n",
      "Requirement already satisfied: filelock in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.3.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from packaging->datasets) (3.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->datasets) (2021.10.8)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->datasets) (3.2)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->datasets) (1.26.7)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
      "Requirement already satisfied: colorama in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from tqdm>=4.62.1->datasets) (0.4.4)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (4.0.2)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (1.2.0)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (1.3.0)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (1.7.2)\n",
      "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (20.3.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (6.0.2)\n",
      "Requirement already satisfied: python-dateutil>=2.7.3 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from pandas->datasets) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2017.3 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from pandas->datasets) (2021.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\saifu\\anaconda3\\lib\\site-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "pip install datasets\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d08d01c7",
   "metadata": {},
   "source": [
    "# 2. Import Libraries:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4e73e0e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import datasets"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af5b11d0",
   "metadata": {},
   "source": [
    "# 3. Load Dataset:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "48fa92e5",
   "metadata": {},
   "source": [
    "### a. load dataset: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "063bc0a6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Reusing dataset math_dataset (C:\\Users\\saifu\\.cache\\huggingface\\datasets\\math_dataset\\algebra__polynomial_roots\\1.0.0\\55e25f9e337098b6760efcf2bbc1c0c9a11d94266b33ad3435bfdcd25aff0377)\n",
      "100%|██████████| 2/2 [00:00<00:00, 53.68it/s]\n"
     ]
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "dataset = load_dataset(\"math_dataset\", \"algebra__polynomial_roots\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d6ca57fd",
   "metadata": {},
   "source": [
    "### b. length of data-set:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d98c161e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4443\n"
     ]
    }
   ],
   "source": [
    "print(len(datasets.list_datasets())) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4a1a40ad",
   "metadata": {},
   "source": [
    "### c. save selected mathematical function to disk: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f10bdb6e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DatasetDict({\n",
       "    train: Dataset({\n",
       "        features: ['question', 'answer'],\n",
       "        num_rows: 1999998\n",
       "    })\n",
       "    test: Dataset({\n",
       "        features: ['question', 'answer'],\n",
       "        num_rows: 10000\n",
       "    })\n",
       "})"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.save_to_disk('algera_poly_roots') # chosen mathematical function: 'Algebra Polynomial Roots'\n",
    "dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2f580f1",
   "metadata": {},
   "source": [
    "# 4. Exploratory Data Analysis:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "339a2c0b",
   "metadata": {},
   "source": [
    "### a. Dataset description:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5a5c1f87",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Mathematics database.\n",
      "\n",
      "This dataset code generates mathematical question and answer pairs,\n",
      "from a range of question types at roughly school-level difficulty.\n",
      "This is designed to test the mathematical learning and algebraic\n",
      "reasoning skills of learning models.\n",
      "\n",
      "Original paper: Analysing Mathematical Reasoning Abilities of Neural Models\n",
      "(Saxton, Grefenstette, Hill, Kohli).\n",
      "\n",
      "Example usage:\n",
      "train_examples, val_examples = datasets.load_dataset(\n",
      "    'math_dataset/arithmetic__mul',\n",
      "    split=['train', 'test'],\n",
      "    as_supervised=True)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "raw_train = dataset[\"train\"]\n",
    "print(raw_train.info.description)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5dcba2a9",
   "metadata": {},
   "source": [
    "### b. check 'train-set' :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2c348871",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'question': \"b'Find j such that -3*j**4/2 - 4908*j**3 + 1407291*j**2/2 - 25725420*j + 70976088 = 0.\\\\n'\", 'answer': \"b'-3411, 3, 68\\\\n'\"}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(raw_train[-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cc1b12d",
   "metadata": {},
   "source": [
    "### c. check 'train-set' feature :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1a7beca1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'question': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None)}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(raw_train.features)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ef5add1",
   "metadata": {},
   "source": [
    "### d. check 'test-set' :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "6c281005",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'question': \"b'Find d, given that 50*d**5/11 - 449140*d**4/11 - 1887262*d**3/11 - 72016*d**2/11 + 1078472*d/11 - 26144 = 0.\\\\n'\", 'answer': \"b'-4, -1, 2/5, 8987\\\\n'\"}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "raw_test = dataset[\"test\"]\n",
    "print(raw_test[-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2d1a79f",
   "metadata": {},
   "source": [
    "### e. check 'test-set' feature :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6560d3b1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'question': Value(dtype='string', id=None), 'answer': Value(dtype='string', id=None)}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(raw_test.features)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "857f8dc2",
   "metadata": {},
   "source": [
    "### f. create pandas data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "85d76694",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "train_df = dataset[\"train\"].to_pandas()\n",
    "test_df = dataset[\"test\"].to_pandas()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9d92dec",
   "metadata": {},
   "source": [
    "### g. summary of 'train' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d916d5a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1999998 entries, 0 to 1999997\n",
      "Data columns (total 2 columns):\n",
      " #   Column    Dtype \n",
      "---  ------    ----- \n",
      " 0   question  object\n",
      " 1   answer    object\n",
      "dtypes: object(2)\n",
      "memory usage: 30.5+ MB\n"
     ]
    }
   ],
   "source": [
    "\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb2f12dd",
   "metadata": {},
   "source": [
    "### h. summary of 'test' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "71c10951",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 2 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   question  10000 non-null  object\n",
      " 1   answer    10000 non-null  object\n",
      "dtypes: object(2)\n",
      "memory usage: 156.4+ KB\n"
     ]
    }
   ],
   "source": [
    "test_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "701609c3",
   "metadata": {},
   "source": [
    "### i. check first 5-rows of 'train' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8ab6694c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>b'Factor 2*k**2/5 - 44598*k/5 + 44596/5.\\n'</td>\n",
       "      <td>b'2*(k - 22298)*(k - 1)/5\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>b'Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = ...</td>\n",
       "      <td>b'-1, 5, 454\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>b'Suppose 27*u**3 - 7329*u**2 = 0. Calculate u...</td>\n",
       "      <td>b'0, 2443/9\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>b'Suppose -5*t**2 - 72845*t - 291300 = 0. Calc...</td>\n",
       "      <td>b'-14565, -4\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>b'Let -2*g**3 - 448774*g**2 - 897542*g - 44877...</td>\n",
       "      <td>b'-224385, -1\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                            question  \\\n",
       "0        b'Factor 2*k**2/5 - 44598*k/5 + 44596/5.\\n'   \n",
       "1  b'Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = ...   \n",
       "2  b'Suppose 27*u**3 - 7329*u**2 = 0. Calculate u...   \n",
       "3  b'Suppose -5*t**2 - 72845*t - 291300 = 0. Calc...   \n",
       "4  b'Let -2*g**3 - 448774*g**2 - 897542*g - 44877...   \n",
       "\n",
       "                         answer  \n",
       "0  b'2*(k - 22298)*(k - 1)/5\\n'  \n",
       "1               b'-1, 5, 454\\n'  \n",
       "2                b'0, 2443/9\\n'  \n",
       "3               b'-14565, -4\\n'  \n",
       "4              b'-224385, -1\\n'  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "train_df.head(5)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e000af5f",
   "metadata": {},
   "source": [
    "### j. take random sample from 'train' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "af76fc14",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>525671</th>\n",
       "      <td>b'Find q, given that -q**5/3 - 11*q**4/3 - 35*...</td>\n",
       "      <td>b'-5, -4, -3, 0, 1\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1102137</th>\n",
       "      <td>b'Factor 4*a**3 - 16*a**2 + 16*a.\\n'</td>\n",
       "      <td>b'4*a*(a - 2)**2\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>937244</th>\n",
       "      <td>b'Factor 4*p**4 - 4*p**3 - 147*p**2 - 142*p - ...</td>\n",
       "      <td>b'(p - 7)*(p + 5)*(2*p + 1)**2\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1107363</th>\n",
       "      <td>b'Suppose -s**3 - 32*s**2/5 + 23*s/5 + 14/5 = ...</td>\n",
       "      <td>b'-7, -2/5, 1\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>541258</th>\n",
       "      <td>b'Determine a so that -2948*a**2/7 + 103156*a/...</td>\n",
       "      <td>b'-6/737, 35\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                  question  \\\n",
       "525671   b'Find q, given that -q**5/3 - 11*q**4/3 - 35*...   \n",
       "1102137               b'Factor 4*a**3 - 16*a**2 + 16*a.\\n'   \n",
       "937244   b'Factor 4*p**4 - 4*p**3 - 147*p**2 - 142*p - ...   \n",
       "1107363  b'Suppose -s**3 - 32*s**2/5 + 23*s/5 + 14/5 = ...   \n",
       "541258   b'Determine a so that -2948*a**2/7 + 103156*a/...   \n",
       "\n",
       "                                    answer  \n",
       "525671               b'-5, -4, -3, 0, 1\\n'  \n",
       "1102137                b'4*a*(a - 2)**2\\n'  \n",
       "937244   b'(p - 7)*(p + 5)*(2*p + 1)**2\\n'  \n",
       "1107363                   b'-7, -2/5, 1\\n'  \n",
       "541258                     b'-6/737, 35\\n'  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "train_df.sample(5) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd87841e",
   "metadata": {},
   "source": [
    "### k. check first 5-rows of 'test' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "490f231f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'</td>\n",
       "      <td>b'-4*p*(p - 6)*(p + 24655)/5\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>b'Determine u, given that 2*u**4 + 88*u**3 - 2...</td>\n",
       "      <td>b'-142, -2, 6, 94\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>b'Let -5*f**4 + 510*f**3 - 13415*f**2 + 20910*...</td>\n",
       "      <td>b'-2, 4, 47, 53\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>b'Solve i**5/8 - 25*i**4/8 - 13*i**3/2 + 175*i...</td>\n",
       "      <td>b'-6, 0, 1, 4, 26\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>b'Solve -3*m**5 - 56385*m**4 - 264628734*m**3 ...</td>\n",
       "      <td>b'-9403, 1, 5\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                            question  \\\n",
       "0   b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'   \n",
       "1  b'Determine u, given that 2*u**4 + 88*u**3 - 2...   \n",
       "2  b'Let -5*f**4 + 510*f**3 - 13415*f**2 + 20910*...   \n",
       "3  b'Solve i**5/8 - 25*i**4/8 - 13*i**3/2 + 175*i...   \n",
       "4  b'Solve -3*m**5 - 56385*m**4 - 264628734*m**3 ...   \n",
       "\n",
       "                            answer  \n",
       "0  b'-4*p*(p - 6)*(p + 24655)/5\\n'  \n",
       "1             b'-142, -2, 6, 94\\n'  \n",
       "2               b'-2, 4, 47, 53\\n'  \n",
       "3             b'-6, 0, 1, 4, 26\\n'  \n",
       "4                 b'-9403, 1, 5\\n'  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2749154",
   "metadata": {},
   "source": [
    "### l. take random sample from 'test' data-frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "58ecff0d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5824</th>\n",
       "      <td>b'Determine v so that -3*v**2/4 + 10155*v/4 + ...</td>\n",
       "      <td>b'-308, 3693\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1549</th>\n",
       "      <td>b'Factor 2*w**3/7 - 4750*w**2/7 + 28414*w/7 + ...</td>\n",
       "      <td>b'2*(w - 2369)*(w - 7)*(w + 1)/7\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7626</th>\n",
       "      <td>b'Factor 2*t**4/3 - 8234*t**3/3 + 178970*t**2 ...</td>\n",
       "      <td>b'2*t*(t - 4051)*(t - 33)**2/3\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3523</th>\n",
       "      <td>b'Determine c so that 3*c**5/4 + 492*c**4 + 35...</td>\n",
       "      <td>b'-417, -9, 0, 187\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6761</th>\n",
       "      <td>b'Factor 4*b**3 + 1751136*b**2 + 1751124*b - 3...</td>\n",
       "      <td>b'4*(b - 1)*(b + 2)*(b + 437783)\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               question  \\\n",
       "5824  b'Determine v so that -3*v**2/4 + 10155*v/4 + ...   \n",
       "1549  b'Factor 2*w**3/7 - 4750*w**2/7 + 28414*w/7 + ...   \n",
       "7626  b'Factor 2*t**4/3 - 8234*t**3/3 + 178970*t**2 ...   \n",
       "3523  b'Determine c so that 3*c**5/4 + 492*c**4 + 35...   \n",
       "6761  b'Factor 4*b**3 + 1751136*b**2 + 1751124*b - 3...   \n",
       "\n",
       "                                   answer  \n",
       "5824                      b'-308, 3693\\n'  \n",
       "1549  b'2*(w - 2369)*(w - 7)*(w + 1)/7\\n'  \n",
       "7626    b'2*t*(t - 4051)*(t - 33)**2/3\\n'  \n",
       "3523                b'-417, -9, 0, 187\\n'  \n",
       "6761  b'4*(b - 1)*(b + 2)*(b + 437783)\\n'  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.sample(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d32dd7d1",
   "metadata": {},
   "source": [
    "### m. compare 'train' & 'test' data-frame by concatenation:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a129703b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "      <th>question</th>\n",
       "      <th>answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>b'Factor 2*k**2/5 - 44598*k/5 + 44596/5.\\n'</td>\n",
       "      <td>b'2*(k - 22298)*(k - 1)/5\\n'</td>\n",
       "      <td>b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'</td>\n",
       "      <td>b'-4*p*(p - 6)*(p + 24655)/5\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>b'Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = ...</td>\n",
       "      <td>b'-1, 5, 454\\n'</td>\n",
       "      <td>b'Determine u, given that 2*u**4 + 88*u**3 - 2...</td>\n",
       "      <td>b'-142, -2, 6, 94\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>b'Suppose 27*u**3 - 7329*u**2 = 0. Calculate u...</td>\n",
       "      <td>b'0, 2443/9\\n'</td>\n",
       "      <td>b'Let -5*f**4 + 510*f**3 - 13415*f**2 + 20910*...</td>\n",
       "      <td>b'-2, 4, 47, 53\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>b'Suppose -5*t**2 - 72845*t - 291300 = 0. Calc...</td>\n",
       "      <td>b'-14565, -4\\n'</td>\n",
       "      <td>b'Solve i**5/8 - 25*i**4/8 - 13*i**3/2 + 175*i...</td>\n",
       "      <td>b'-6, 0, 1, 4, 26\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>b'Let -2*g**3 - 448774*g**2 - 897542*g - 44877...</td>\n",
       "      <td>b'-224385, -1\\n'</td>\n",
       "      <td>b'Solve -3*m**5 - 56385*m**4 - 264628734*m**3 ...</td>\n",
       "      <td>b'-9403, 1, 5\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                            question  \\\n",
       "0        b'Factor 2*k**2/5 - 44598*k/5 + 44596/5.\\n'   \n",
       "1  b'Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = ...   \n",
       "2  b'Suppose 27*u**3 - 7329*u**2 = 0. Calculate u...   \n",
       "3  b'Suppose -5*t**2 - 72845*t - 291300 = 0. Calc...   \n",
       "4  b'Let -2*g**3 - 448774*g**2 - 897542*g - 44877...   \n",
       "\n",
       "                         answer  \\\n",
       "0  b'2*(k - 22298)*(k - 1)/5\\n'   \n",
       "1               b'-1, 5, 454\\n'   \n",
       "2                b'0, 2443/9\\n'   \n",
       "3               b'-14565, -4\\n'   \n",
       "4              b'-224385, -1\\n'   \n",
       "\n",
       "                                            question  \\\n",
       "0   b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'   \n",
       "1  b'Determine u, given that 2*u**4 + 88*u**3 - 2...   \n",
       "2  b'Let -5*f**4 + 510*f**3 - 13415*f**2 + 20910*...   \n",
       "3  b'Solve i**5/8 - 25*i**4/8 - 13*i**3/2 + 175*i...   \n",
       "4  b'Solve -3*m**5 - 56385*m**4 - 264628734*m**3 ...   \n",
       "\n",
       "                            answer  \n",
       "0  b'-4*p*(p - 6)*(p + 24655)/5\\n'  \n",
       "1             b'-142, -2, 6, 94\\n'  \n",
       "2               b'-2, 4, 47, 53\\n'  \n",
       "3             b'-6, 0, 1, 4, 26\\n'  \n",
       "4                 b'-9403, 1, 5\\n'  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.concat([train_df.head(5),test_df.head(5)], axis=1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff4dcf1f",
   "metadata": {},
   "source": [
    "### n. check missing values of 'train' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "75045025",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>question</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>answer</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      index  0\n",
       "0  question  0\n",
       "1    answer  0"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "train_df.isnull().sum().to_frame().reset_index()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a265b2b8",
   "metadata": {},
   "source": [
    "### o. check missing values of 'test' data-frame: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7dc7191d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>question</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>answer</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      index  0\n",
       "0  question  0\n",
       "1    answer  0"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.isnull().sum().to_frame().reset_index()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbb78bef",
   "metadata": {},
   "source": [
    "### p. drop duplicates of 'train' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "dc1e79f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>question</td>\n",
       "      <td>1999998</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>answer</td>\n",
       "      <td>1999998</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      index        0\n",
       "0  question  1999998\n",
       "1    answer  1999998"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.drop_duplicates()\n",
    "train_count = train_df.count().to_frame().reset_index()\n",
    "#train_count.plot(kind='bar').legend('count')\n",
    "train_count"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed742a1b",
   "metadata": {},
   "source": [
    "### q. drop duplicates of 'test' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "ea069720",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>question</td>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>answer</td>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      index      0\n",
       "0  question  10000\n",
       "1    answer  10000"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.drop_duplicates()\n",
    "test_count = test_df.count().to_frame().reset_index()\n",
    "#test_count.plot(kind='bar').legend('count')\n",
    "test_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2f10944d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pd.concat([train_count, test_count])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2376c650",
   "metadata": {},
   "source": [
    "### r. rename columns of 'train' data-frame: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "c1337167",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>train_question</th>\n",
       "      <th>train_answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>b'Factor 2*k**2/5 - 44598*k/5 + 44596/5.\\n'</td>\n",
       "      <td>b'2*(k - 22298)*(k - 1)/5\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>b'Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = ...</td>\n",
       "      <td>b'-1, 5, 454\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>b'Suppose 27*u**3 - 7329*u**2 = 0. Calculate u...</td>\n",
       "      <td>b'0, 2443/9\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>b'Suppose -5*t**2 - 72845*t - 291300 = 0. Calc...</td>\n",
       "      <td>b'-14565, -4\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>b'Let -2*g**3 - 448774*g**2 - 897542*g - 44877...</td>\n",
       "      <td>b'-224385, -1\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                      train_question  \\\n",
       "0        b'Factor 2*k**2/5 - 44598*k/5 + 44596/5.\\n'   \n",
       "1  b'Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = ...   \n",
       "2  b'Suppose 27*u**3 - 7329*u**2 = 0. Calculate u...   \n",
       "3  b'Suppose -5*t**2 - 72845*t - 291300 = 0. Calc...   \n",
       "4  b'Let -2*g**3 - 448774*g**2 - 897542*g - 44877...   \n",
       "\n",
       "                   train_answer  \n",
       "0  b'2*(k - 22298)*(k - 1)/5\\n'  \n",
       "1               b'-1, 5, 454\\n'  \n",
       "2                b'0, 2443/9\\n'  \n",
       "3               b'-14565, -4\\n'  \n",
       "4              b'-224385, -1\\n'  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.rename(columns= {'question':'train_question', 'answer':'train_answer'}, inplace=True)\n",
    "train_df.head(5)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "168f91c0",
   "metadata": {},
   "source": [
    "### s. rename columns of 'test' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7717e12d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>test_question</th>\n",
       "      <th>test_answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'</td>\n",
       "      <td>b'-4*p*(p - 6)*(p + 24655)/5\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>b'Determine u, given that 2*u**4 + 88*u**3 - 2...</td>\n",
       "      <td>b'-142, -2, 6, 94\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>b'Let -5*f**4 + 510*f**3 - 13415*f**2 + 20910*...</td>\n",
       "      <td>b'-2, 4, 47, 53\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>b'Solve i**5/8 - 25*i**4/8 - 13*i**3/2 + 175*i...</td>\n",
       "      <td>b'-6, 0, 1, 4, 26\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>b'Solve -3*m**5 - 56385*m**4 - 264628734*m**3 ...</td>\n",
       "      <td>b'-9403, 1, 5\\n'</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                       test_question  \\\n",
       "0   b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'   \n",
       "1  b'Determine u, given that 2*u**4 + 88*u**3 - 2...   \n",
       "2  b'Let -5*f**4 + 510*f**3 - 13415*f**2 + 20910*...   \n",
       "3  b'Solve i**5/8 - 25*i**4/8 - 13*i**3/2 + 175*i...   \n",
       "4  b'Solve -3*m**5 - 56385*m**4 - 264628734*m**3 ...   \n",
       "\n",
       "                       test_answer  \n",
       "0  b'-4*p*(p - 6)*(p + 24655)/5\\n'  \n",
       "1             b'-142, -2, 6, 94\\n'  \n",
       "2               b'-2, 4, 47, 53\\n'  \n",
       "3             b'-6, 0, 1, 4, 26\\n'  \n",
       "4                 b'-9403, 1, 5\\n'  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.rename(columns= {'question':'test_question', 'answer':'test_answer'}, inplace=True)\n",
    "test_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c62b2a4",
   "metadata": {},
   "source": [
    "### t. generate descriptive statistics 'train' data-frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "89765f80",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>train_question</th>\n",
       "      <th>train_answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1999998</td>\n",
       "      <td>1999998</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>1997171</td>\n",
       "      <td>1693252</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>b'Factor 2*n**2 + 4*n.\\n'</td>\n",
       "      <td>b'-1, 0, 1\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>4</td>\n",
       "      <td>1264</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   train_question   train_answer\n",
       "count                     1999998        1999998\n",
       "unique                    1997171        1693252\n",
       "top     b'Factor 2*n**2 + 4*n.\\n'  b'-1, 0, 1\\n'\n",
       "freq                            4           1264"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a73b0fac",
   "metadata": {},
   "source": [
    "### u. generate descriptive statistics 'test' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "f2e090e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>test_question</th>\n",
       "      <th>test_answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10000</td>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>10000</td>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'</td>\n",
       "      <td>b'-4*p*(p - 6)*(p + 24655)/5\\n'</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                           test_question  \\\n",
       "count                                              10000   \n",
       "unique                                             10000   \n",
       "top     b'Factor -4*p**3/5 - 98596*p**2/5 + 118344*p.\\n'   \n",
       "freq                                                   1   \n",
       "\n",
       "                            test_answer  \n",
       "count                             10000  \n",
       "unique                            10000  \n",
       "top     b'-4*p*(p - 6)*(p + 24655)/5\\n'  \n",
       "freq                                  1  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c9a04c9d",
   "metadata": {},
   "source": [
    "### v. count distinct elements in 'train' data-frame:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "01496202",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>train_question</th>\n",
       "      <td>1997171</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>train_answer</th>\n",
       "      <td>1693252</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      0\n",
       "train_question  1997171\n",
       "train_answer    1693252"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_unique = train_df.nunique().to_frame()\n",
    "train_unique"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e05f2d7",
   "metadata": {},
   "source": [
    "### w. count distinct elements in 'test' data-frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "0b67066a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>test_question</th>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>test_answer</th>\n",
       "      <td>10000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   0\n",
       "test_question  10000\n",
       "test_answer    10000"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_unique = test_df.nunique().to_frame()\n",
    "test_unique"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "196a8ee4",
   "metadata": {},
   "source": [
    "### x. visualization: compare unique values 'train' & 'test' data-frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "22d280ef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x22387ef5250>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1440x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes=plt.subplots(nrows=1, ncols=2)\n",
    "train_unique.plot(ax=axes[0], kind='bar', title='train unique', figsize=(20,6)).legend('count')\n",
    "test_unique.plot(ax=axes[1], kind='bar', title='test unique', figsize=(20,6)).legend('count')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f07e79f",
   "metadata": {},
   "source": [
    "### y. save 'train' data-frame as 'csv.' :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "70d911e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df.to_csv('Train.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea7b4818",
   "metadata": {},
   "source": [
    "### z. save 'test' data-frame as 'csv'. :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "365978a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_df.to_csv('Test.csv')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}