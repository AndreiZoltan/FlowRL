

Этот репозиторий основан на репозитории ByteDance (https://github.com/bytedance/FlowRL).


## Getting Started

1. **Setup Conda Environment:**
    Create an environment with
    ```bash
    conda create -n flowrl python=3.11
    ```

2. **Clone this Repository:**
    ```bash
    git clone https://github.com/AndreiZoltan/FlowRL.git
    cd FlowRL
    ```

3. **Install FlowRL Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Training Examples:**
    - Run a single training instance:
        ```bash
        python3 main.py --domain dog --task run --method rk4 --path Affine
        ```
For linear path use `--path CondOT` for variance preserving use `--path Affine`.
Methods choose among `midpoint`, `rk4` and `implicit_adams`.
## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
