
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('get-started.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('get-started.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Convert 'baths' column to numeric with errors='coerce'
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

    # Convert input data to numeric types
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
    





































"""from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('get-started.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('get-started.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Convert 'baths' column to numeric with errors='coerce'
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

    # Convert input data to numeric types
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    app.run(debug=True, port=5000)"""




































































































"""import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    try:
        data = [float(x) for x in request.form.values()]
        final_features = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)

        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction[0]:,.2f}')
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter valid numeric values.')

if __name__ == "__main__":
    app.run(debug=True)"""
    
"""from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('final_dataset.csv')

# Define features and target
X = data[['beds', 'baths', 'size', 'zip_code']]
y = data['price']  # Assuming 'price' is the target column

# Create a pipeline with normalization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Fit the pipeline
pipeline.fit(X, y)

# Save the pipeline
with open('RidgeModel.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

@app.route('/')
def index():
    # Get unique values for dropdown menu
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())
    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Check if any of the inputs are None
        if None in [bedrooms, bathrooms, size, zipcode]:
            return "Error: One or more inputs are missing.", 400

        # Convert input data to appropriate types
        bedrooms = int(bedrooms)
        bathrooms = float(bathrooms)
        size = float(size)
        zipcode = int(zipcode)

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]], columns=['beds', 'baths', 'size', 'zip_code'])

        print("Input Data:")
        print(input_data)

        # Handle unknown categories in the input data
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                print(f"Unknown categories in {column}: {unknown_categories}")
                # Handle unknown categories (e.g., replace with a default value)
                input_data[column] = input_data[column].replace(list(unknown_categories), data[column].mode()[0])

        print("Processed Input Data:")
        print(input_data)

        # Load the pre-trained model pipeline
        with open('RidgeModel.pkl', 'rb') as f:
            model = pickle.load(f)

        # Predict the price using the loaded model
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction:,.2f}')
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=8001)"""
 




"""import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('final_dataset.csv')

# Load the pre-trained model pipeline
with open('RidgeModel.pkl', 'rb') as f:
    model = pickle.load(f)
print(type(model))

@app.route('/')
def index():
    # Get unique values for dropdown menu
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())
    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Check if any of the inputs are None
        if None in [bedrooms, bathrooms, size, zipcode]:
            return "Error: One or more inputs are missing.", 400

        # Convert input data to appropriate types
        bedrooms = int(bedrooms)
        bathrooms = float(bathrooms)
        size = float(size)
        zipcode = int(zipcode)

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]], columns=['beds', 'baths', 'size', 'zip_code'])

        print("Input Data:")
        print(input_data)

        # Handle unknown categories in the input data
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                print(f"Unknown categories in {column}: {unknown_categories}")
                # Handle unknown categories (e.g., replace with a default value)
                input_data[column] = input_data[column].replace(list(unknown_categories), data[column].mode()[0])

        print("Processed Input Data:")
        print(input_data)

        # Check if model has a transform method and transform the data
        if hasattr(model, 'transform'):
            transformed_data = model.transform(input_data)
        else:
            transformed_data = input_data

        # Predict the price using the loaded model
        prediction = model.predict(transformed_data)[0]

        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction:,.2f}')
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=8001)"""



"""from os import pipe
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

import importlib.util


#from imp import load_module # type: ignore
#from imp import load_module

app = Flask(__name__)

# Load the dataset and the pre-trained model
data = pd.read_csv('final_dataset.csv')
# Load the pipeline
with open('RidgeModel.pkl', 'rb') as f:
    model = pickle.load(f)
print(type(model))

@app.route('/')
def index():
    # Get unique values for dropdown menu
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())
    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Check if any of the inputs are None
        if None in [bedrooms, bathrooms, size, zipcode]:
            return "Error: One or more inputs are missing.", 400

        # Convert input data to appropriate types
        bedrooms = int(bedrooms)
        bathrooms = float(bathrooms)
        size = float(size)
        zipcode = int(zipcode)

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]], columns=['beds', 'baths', 'size', 'zip_code'])

        print("Input Data:")
        print(input_data)

        # Handle unknown categories in the input data
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                print(f"Unknown categories in {column}: {unknown_categories}")
                # Handle unknown categories (e.g., replace with a default value)
                input_data[column] = input_data[column].replace(list(unknown_categories), data[column].mode()[0])

        print("Processed Input Data:")
        print(input_data)

        # Load your trained model from a file (replace with your loading logic)
       # model = load_module('RidgeModel.pkl')

        # Use the loaded model for prediction
        prediction = model.predict(input_data)[0]

        # Predict the price using the loaded model

        prediction = pipe.predict(input_data)[0]

        return str(prediction)
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=8001)"""












































