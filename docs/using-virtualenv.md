# Using virtualenv to install pandas

1. Install virtualenv:

    ```bash
    pip install virtualenv
    ```

1. Create a virtual environment:

    ```bash
    virtualenv -p python3.12.2 venv
    ```

2. Activate the virtual environment:
  
    ```bash
    source venv/bin/activate
    ```

3. Upgrade pip (optional but recommended):
  
    ```bash
    pip install --upgrade pip
    ```

4. Install required packages, e.g., pandas:
  
    ```bash
    pip install pandas
    ```