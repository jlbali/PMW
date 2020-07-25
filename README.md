# Python Data Science Template



sudo apt-get install python3-dev
sudo apt-get install cmake (para xgboost.. igual da problemas)


## Workflow:

### Instalación:

- Necesario tener el pip3 y recomendable el python3.6 al menos.
- Ejecutar ./venv_create.sh
- Ejecutar source ./venv_activate.sh
- Ejecutar ./venv_install.sh

### Activar el entorno

- Ejecutar source ./venv_activate.sh

Siempre asegurarse que en el prompt aparezca (env).


### PyCharm

- Ejecutar el PyCharm y crear nuevo proyecto. Indicarle que use Existing Interpreter, marcarle Virtualenv y marcarle el path dentro de env/bin/python3.



### Notas

- A veces la activación del environment falla sin decir nada. Debe aparecer un (env) en el prompt. Si no aparece, ir manualmente al directorio env/bin y ahí hacer el source activate

- Si falla el install.sh, usar el altInstall.sh...

- Lección 5 y 6 de Spark podrían complementarse. En la lección 5 se usa SQL, pero de una forma tosca se construyen los dataframes (a partir de RDDs). En la lección 6 se levantan los dataframes de forma más natural, aunque no tiene consultas SQL.

- SparkContext es para RDDs en general... SparkSession es para DataFrames.


