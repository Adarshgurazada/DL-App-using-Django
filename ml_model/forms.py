from django import forms

class IrisForm(forms.Form):
    sepal_length = forms.FloatField(label='Sepal Length')
    sepal_width = forms.FloatField(label='Sepal Width')
    petal_length = forms.FloatField(label='Petal Length')
    petal_width = forms.FloatField(label='Petal Width')
