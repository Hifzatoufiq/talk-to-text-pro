python -m venv .venv
source .venv/bin/activate # mac/linux
.venv\Scripts\activate # windows
pip install -r requirements.txt
cp .env.example .env
# then edit .env to add your keys
flask run
