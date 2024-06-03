from django.shortcuts import render
from django.http import HttpResponse
from .forms import IrisForm 
from .ml_functions import preprocess_data 
import pickle
from django.contrib.auth.decorators import login_required

@login_required
def ml_model_view(request):
    if request.method == 'POST':
        form = IrisForm(request.POST)  
        if form.is_valid():
            # Get input values from the form
            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']

            # Load the trained model
            with open('ml_model/iris_model.pkl', 'rb') as file:
                model = pickle.load(file)

            # Make prediction based on user input
            new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = model.predict(new_data)
            species_names = ['Setosa', 'Versicolor', 'Virginica']
            predicted_species = species_names[prediction[0]]

            # Prepare context to pass to the template
            context = {
                'form': form,
                'predicted_species': predicted_species,
            }
            return render(request, 'ml_model.html', context)
    else:
        form = IrisForm() 

    context = {
        'form': form,
    }
    return render(request, 'ml_model.html', context)
