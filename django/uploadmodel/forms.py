from django import forms
from utils.constant import NETWORK_MODEL_API, NETWORK_API, NETWORK_MODEL_BY_NETWORK_API
from conf.settings import NNDB_SERVER
from django.utils.translation import gettext_lazy as _
import requests
import logging
import os
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator

logger = logging.getLogger(__name__)

def validate_integer(value):
    if value is not None and not isinstance(value, int):
        raise forms.ValidationError('Please enter a valid integer value.')


def validate_folder_path(value):
    if not os.path.exists(value) or not os.path.isdir(value):
        raise ValidationError('Invalid folder path.')

class UploadWandModelForm(forms.Form):
    
    wand_ip = forms.CharField(
        label='Wand IP',
        widget=forms.TextInput(
        attrs={
            'class': 'form-control', 
            'placeholder': 
            'Example: cm4-3.local'
            }),
        required=False)

    dataset_size = forms.IntegerField(
        label='Dataset Size',
        widget=forms.NumberInput(
        attrs={'class': 'form-control', 'placeholder': 'Example: 5'}),
        required=False,
        validators=[validate_integer]
        )

    flash = forms.DecimalField(
        label='Flash',
        widget=forms.NumberInput(
        attrs={'class': 'form-control', 'placeholder': 'Example: 0 or 1'}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0'),
            MaxValueValidator(1, message='Value must be less than or equal to 1'),
        ],
        initial=0.5
        )
    
    dias = forms.DecimalField(
        label='Dias',
        widget=forms.NumberInput(
        attrs={'class': 'form-control', 'placeholder': 'Example: 0 or 1'}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0'),
            MaxValueValidator(1, message='Value must be less than or equal to 1'),
        ],
        initial=1
        )

    input_folder = forms.CharField(
        label=_('Input Folder'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'}),
        validators=[validate_folder_path], required=True,
        initial='/danbots/data2/data/wand/'
    )

    uploaded_model = forms.ChoiceField(
        label=_('Target Upload Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    depth_model = forms.ChoiceField(
        label=_('Depth Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    wrap_model = forms.ChoiceField(
        label=_('Wrap Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    lum_model = forms.ChoiceField(
        label=_('Lum Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    opa_model = forms.ChoiceField(
        label=_('Opa Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    batch_name = forms.CharField(
        label=_('Batch Name'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    
    # start = forms.CharField(
    #     label=_('Start'),
    #     max_length=100,
    #     widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    # )
    # stop = forms.CharField(
    #     label=_('Stop'),
    #     max_length=100,
    #     widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    # )
    # step = forms.CharField(
    #     label=_('Step'),
    #     max_length=100,
    #     widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    # )
   
    DEPTH_COMPARISON_CHOICES = [
        ('', 'Please select one type'),
        ('plane', 'Plane'),
        ('sphere', 'Sphere'),
        ('stl', 'STL'),
    ]
    depth_comparison = forms.ChoiceField(
            label=_('Depth comparison type'),
            choices=DEPTH_COMPARISON_CHOICES,
            widget=forms.Select(attrs={'class': 'form-control'})
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_data = self.fetch_model_data()
        self.depth_model_data = self.fetch_model_data(parmater="depth")
        self.wrap_model_data = self.fetch_model_data(parmater="wrap")
        self.lum_model_data = self.fetch_model_data(parmater="lum")
        self.opa_model_data = self.fetch_model_data(parmater="opa")
        self.fields['uploaded_model'].choices = self.get_model_choices('id')
        self.fields['depth_model'].choices = self.get_depth_model_choices('model_path', True)
        self.fields['wrap_model'].choices = self.get_wrap_model_choices('model_path', True)
        self.fields['lum_model'].choices = self.get_lum_model_choices('model_path', True)
        self.fields['opa_model'].choices = self.get_opa_model_choices('model_path', True)

    def fetch_model_data(self, parmater=None):
        if parmater:
            if parmater == "depth":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=depth"
            elif parmater == "wrap":
                 api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=wrap"
            elif parmater == "lum":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=lum"
            elif parmater == "opa":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=opa"
            else:
                api_url = NNDB_SERVER + NETWORK_MODEL_API
        else:
            api_url = NNDB_SERVER + NETWORK_MODEL_API
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # Raise exception if response status code is not 2xx
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle the exception, e.g., log the error or show a user-friendly message
            logger.exception("Error fetching network models: %s", e)
            return []
    
    def get_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a target model {name}')]
        for model in self.model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_depth_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a depth model {name}')]
        for model in self.depth_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_wrap_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a wrap model {name}')]
        for model in self.wrap_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_lum_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a lum model {name}')]
        for model in self.lum_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    
    def get_opa_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a opa model {name}')]
        for model in self.opa_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    

class UploadSimulationModelForm(forms.Form):
    input_folder = forms.CharField(
        label=_('Input Folder'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'}),
        validators=[validate_folder_path], required=True,
    )

    uploaded_model = forms.ChoiceField(
        label=_('Target Upload Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    depth_model = forms.ChoiceField(
        label=_('Depth Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    wrap_model = forms.ChoiceField(
        label=_('Wrap Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    lum_model = forms.ChoiceField(
        label=_('Lum Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    opa_model = forms.ChoiceField(
        label=_('Opa Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    batch_name = forms.CharField(
        label=_('Batch Name'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'})
    )

    start = forms.CharField(
        label=_('Start'),
        max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )
    stop = forms.CharField(
        label=_('Stop'),
        max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )
    step = forms.CharField(
        label=_('Step'),
        max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_data = self.fetch_model_data()
        self.depth_model_data = self.fetch_model_data(parmater="depth")
        self.wrap_model_data = self.fetch_model_data(parmater="wrap")
        self.lum_model_data = self.fetch_model_data(parmater="lum")
        self.opa_model_data = self.fetch_model_data(parmater="opa")

        self.fields['uploaded_model'].choices = self.get_model_choices('id')
        self.fields['depth_model'].choices = self.get_depth_model_choices('model_path', True)
        self.fields['wrap_model'].choices = self.get_wrap_model_choices('model_path', True)
        self.fields['lum_model'].choices = self.get_lum_model_choices('model_path', True)
        self.fields['opa_model'].choices = self.get_opa_model_choices('model_path', True)

    def fetch_model_data(self, parmater=None):
        if parmater:
            if parmater == "depth":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=depth"
            elif parmater == "wrap":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=wrap"
            elif parmater == "lum":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=lum"
            elif parmater == "opa":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=opa"
            else:
                api_url = NNDB_SERVER + NETWORK_MODEL_API
        else:
            api_url = NNDB_SERVER + NETWORK_MODEL_API
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # Raise exception if response status code is not 2xx
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle the exception, e.g., log the error or show a user-friendly message
            logger.exception("Error fetching network models: %s", e)
            return []

    def get_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a target model {name}')]
        for model in self.model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices

    def get_depth_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a depth model {name}')]
        for model in self.depth_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices

    def get_wrap_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a wrap model {name}')]
        for model in self.wrap_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices

    def get_lum_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a lum model {name}')]
        for model in self.lum_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    
    def get_opa_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a opa model {name}')]
        for model in self.opa_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices


class UploadStitchingSimulationModelForm(forms.Form):
    
    input_folder = forms.CharField(
        label=_('Input Folder'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'}),
        validators=[validate_folder_path], required=True,
    )

    uploaded_model = forms.ChoiceField(
        label=_('Target Upload Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    depth_model = forms.ChoiceField(
        label=_('Depth Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    wrap_model = forms.ChoiceField(
        label=_('Wrap Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    lum_model = forms.ChoiceField(
        label=_('Lum Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    opa_model = forms.ChoiceField(
        label=_('Opa Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    batch_name = forms.CharField(
        label=_('Batch Name'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'})
    )

    start = forms.CharField(
        label=_('Start'),
        max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )
    stop = forms.CharField(
        label=_('Stop'),
        max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )
    step = forms.CharField(
        label=_('Step'),
        max_length=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )

    nb_points = forms.CharField(
        label=_('NB.Points'),
        max_length=100,
        initial=25,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1},)
    )

    radius = forms.FloatField(
        label=_('Radius'),
        initial=0.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )
 
    voxel_size = forms.FloatField(
        label=_('Voxel Size'),
        initial=0.02,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_data = self.fetch_model_data()
        self.depth_model_data = self.fetch_model_data(parmater="depth")
        self.wrap_model_data = self.fetch_model_data(parmater="wrap")
        self.lum_model_data = self.fetch_model_data(parmater="lum")
        self.opa_model_data = self.fetch_model_data(parmater="opa")
        self.fields['uploaded_model'].choices = self.get_model_choices('id')
        self.fields['depth_model'].choices = self.get_depth_model_choices('model_path', True)
        self.fields['wrap_model'].choices = self.get_wrap_model_choices('model_path', True)
        self.fields['lum_model'].choices = self.get_lum_model_choices('model_path', True)
        self.fields['opa_model'].choices = self.get_opa_model_choices('model_path', True)


    def fetch_model_data(self, parmater=None):
        if parmater:
            if parmater == "depth":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=depth"
            elif parmater == "wrap":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=wrap"
            elif parmater == "lum":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=lum"
            elif parmater == "opa":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=opa"
            else:
                api_url = NNDB_SERVER + NETWORK_MODEL_API
        else:
            api_url = NNDB_SERVER + NETWORK_MODEL_API
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # Raise exception if response status code is not 2xx
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle the exception, e.g., log the error or show a user-friendly message
            logger.exception("Error fetching network models: %s", e)
            return []
    
    def get_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a target model {name}')]
        for model in self.model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_depth_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a depth model {name}')]
        for model in self.depth_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_wrap_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a wrap model {name}')]
        for model in self.wrap_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    
    def get_lum_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a lum model {name}')]
        for model in self.lum_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
        
    def get_opa_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a opa model {name}')]
        for model in self.opa_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices

class InferUploadedModelForm(forms.Form):
    net = forms.ChoiceField(choices=[], label="Network", widget=forms.Select(attrs={'class': 'form-control'}))
    model = forms.ChoiceField(choices=[], label='Model', required=True, widget=forms.Select(attrs={'class': 'form-control'}))
    is_currently_used = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'checked': 'checked'}))

    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request', None)
        super(InferUploadedModelForm, self).__init__(*args, **kwargs)
        
        # Initialize network choices
        self.fields['net'].choices = self.fetch_network_data()
        
        # Update model choices based on net selection in POST request
        if self.request and self.request.method == 'POST':
            net_id = self.request.POST.get('net')
            self.fields['model'].choices = self.get_models_for_net(net_id)

    def fetch_network_data(self):
        api_url = NNDB_SERVER + NETWORK_API
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            network_data = response.json()
            network_choices = [(None, "Please select a network")] + [(net['id'], net['name']) for net in network_data]
            return network_choices
        except requests.exceptions.RequestException as e:
            # Handle the exception, e.g., log the error
            print(f"Failed to fetch networks: {e}")
            return [(None, "Please select a network")]

    def get_models_for_net(self, net_id):
        api_url = f"{NNDB_SERVER}{NETWORK_MODEL_BY_NETWORK_API}{net_id}/"
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # This will raise an error for 4xx/5xx responses
            models_data = response.json()
            return [(model['id'], model['model_path']) for model in models_data if 'id' in model and 'model_path' in model]
        except requests.exceptions.RequestException as e:
            # Log the error or handle it as appropriate
            print(f"Failed to fetch models for network {net_id}: {e}")
            return []  # Return an empty list if the request fails
    
    def clean_model(self):
        model_id = self.cleaned_data.get('model')
        # Add actual validation logic as needed
        if not model_id:
            raise ValidationError("Please select a valid model")
        return model_id
class SimSegmentStitchingPointCloudForm(forms.Form):
    uploaded_model = forms.ChoiceField(
        label=_('Target Upload Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )

    stitched_file = forms.MultipleChoiceField(choices=[], label='Stitched File', required=True, widget=forms.SelectMultiple(attrs={'class': 'form-control'}))
    
    batch_name = forms.CharField(
        label=_('Batch Name'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    nb_points = forms.CharField(
        label=_('NB.Points'),
        max_length=100,
        initial=25,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required', 'step': 1})
    )
    radius = forms.FloatField(
        label=_('Radius'),
        initial=0.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    voxel_size = forms.FloatField(
        label=_('Voxel Size'),
        initial=0.02,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )

    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request', None)
        super(SimSegmentStitchingPointCloudForm, self).__init__(*args, **kwargs)
        self.model_data = self.fetch_model_data()
        self.fields['uploaded_model'].choices = self.get_model_choices('id')

        if self.request:
            uploaded_model = self.request.POST.get('uploaded_model') or self.request.GET.get('uploaded_model')
            if uploaded_model:
                self.fields['stitched_file'].choices = self.fetch_stitched_file_choices(uploaded_model)
                # Debug print
                # print("Stitched File Choices:", self.fields['stitched_file'].choices)

    def fetch_model_data(self, parameter=None):
        api_url = NNDB_SERVER + NETWORK_MODEL_API
        if parameter:
            api_url += f"?type={parameter}"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.exception("Error fetching network models: %s", e)
            return []

    def fetch_stitched_file_choices(self, uploaded_model_id):
        api_url = f"{NNDB_SERVER}apis/v1/model/{uploaded_model_id}/batch-name-stitching-simulation-point-cloud"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            stitched_files = response.json()
            return [(stitched_file['output_folder'], stitched_file['name']) for stitched_file in stitched_files if 'output_folder' in stitched_file and 'name' in stitched_file]
        except requests.exceptions.RequestException as e:
            logger.exception("Error fetching stitched files: %s", e)
            return []

    def clean_stitched_file(self):
        stitched_file = self.cleaned_data.get('stitched_file')
        if not stitched_file:
            raise ValidationError("Please select a valid stitched file")
        return stitched_file

    def get_model_choices(self, field_name, include_id=False):
        name = 'ID' if field_name == 'id' else 'path'
        choices = [('', f'Please select a target model {name}')]
        for model in self.model_data:
            choice_label = f"{model[field_name]}" if include_id else model[field_name]
            choices.append((model[field_name], choice_label))
        return choices

class UploadWandXYZDisplacementModelForm(forms.Form):
    
    wand_ip = forms.CharField(
        label='Wand IP',
        widget=forms.TextInput(
        attrs={
            'class': 'form-control', 
            'placeholder': 
            'Example: cm4-3.local'
            }),
        required=True)

    dataset_size = forms.IntegerField(
        label='Dataset Size',
        widget=forms.NumberInput(
        attrs={'class': 'form-control', 'placeholder': 'Example: 5'}),
        required=True,
        validators=[validate_integer]
        )
    x_displacement = forms.DecimalField(
        label='X Displacement(mm)',
        widget=forms.NumberInput(
        attrs={'class': 'form-control'}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0 mm'),
            MaxValueValidator(1000, message='Value must be less than or equal to 1000 mm'),
        ],
        initial=0
        )
    y_displacement = forms.DecimalField(
        label='Y Displacement(mm)',
        widget=forms.NumberInput(
        attrs={'class': 'form-control',}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0 mm'),
            MaxValueValidator(1000, message='Value must be less than or equal to 1000 mm'),
        ],
        initial=0
        )
    z_displacement = forms.DecimalField(
        label='Z Displacement(mm)',
        widget=forms.NumberInput(
        attrs={'class': 'form-control',}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0 mm'),
            MaxValueValidator(1000, message='Value must be less than or equal to 1000 mm'),
        ],
        initial=1
        )

    flash = forms.DecimalField(
        label='Flash',
        widget=forms.NumberInput(
        attrs={'class': 'form-control', 'placeholder': 'Example: 0 or 1'}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0'),
            MaxValueValidator(1, message='Value must be less than or equal to 1'),
        ],
        initial=0.5
        )
    
    dias = forms.DecimalField(
        label='Dias',
        widget=forms.NumberInput(
        attrs={'class': 'form-control', 'placeholder': 'Example: 0 or 1'}),
        required=False,
        validators=[
            MinValueValidator(0, message='Value must be greater than or equal to 0'),
            MaxValueValidator(1, message='Value must be less than or equal to 1'),
        ],
        initial=1
        )

    input_folder = forms.CharField(
        label=_('Input Folder'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'}),
        validators=[validate_folder_path], required=True,
        initial='/danbots/data2/data/wand/'
    )

    uploaded_model = forms.ChoiceField(
        label=_('Target Upload Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    depth_model = forms.ChoiceField(
        label=_('Depth Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    wrap_model = forms.ChoiceField(
        label=_('Wrap Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    lum_model = forms.ChoiceField(
        label=_('Lum Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    opa_model = forms.ChoiceField(
        label=_('Opa Model'),
        widget=forms.Select(attrs={'class': 'form-control', 'required': 'required'})
    )
    batch_name = forms.CharField(
        label=_('Batch Name'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    
    DEPTH_COMPARISON_CHOICES = [
        ('', 'Please select one type'),
        ('plane', 'Plane'),
        ('sphere', 'Sphere'),
        ('stl', 'STL'),
    ]
    depth_comparison = forms.ChoiceField(
            label=_('Depth comparison type'),
            choices=DEPTH_COMPARISON_CHOICES,
            widget=forms.Select(attrs={'class': 'form-control'})
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_data = self.fetch_model_data()
        self.depth_model_data = self.fetch_model_data(parmater="depth")
        self.wrap_model_data = self.fetch_model_data(parmater="wrap")
        self.lum_model_data = self.fetch_model_data(parmater="lum")
        self.opa_model_data = self.fetch_model_data(parmater="opa")
        self.fields['uploaded_model'].choices = self.get_model_choices('id')
        self.fields['depth_model'].choices = self.get_depth_model_choices('model_path', True)
        self.fields['wrap_model'].choices = self.get_wrap_model_choices('model_path', True)
        self.fields['lum_model'].choices = self.get_lum_model_choices('model_path', True)
        self.fields['opa_model'].choices = self.get_opa_model_choices('model_path', True)


    def fetch_model_data(self, parmater=None):
        if parmater:
            if parmater == "depth":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=depth"
            elif parmater == "wrap":
                 api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=wrap"
            elif parmater == "lum":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=lum"
            elif parmater == "opa":
                api_url = NNDB_SERVER + NETWORK_MODEL_API + "?type=opa"
            else:
                api_url = NNDB_SERVER + NETWORK_MODEL_API
        else:
            api_url = NNDB_SERVER + NETWORK_MODEL_API
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # Raise exception if response status code is not 2xx
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle the exception, e.g., log the error or show a user-friendly message
            logger.exception("Error fetching network models: %s", e)
            return []
    
    def get_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a target model {name}')]
        for model in self.model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_depth_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a depth model {name}')]
        for model in self.depth_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_wrap_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a wrap model {name}')]
        for model in self.wrap_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_lum_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a lum model {name}')]
        for model in self.lum_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices
    def get_opa_model_choices(self, field_name, include_id=False):
        if field_name == 'id':
            name = 'ID'
        else:
            name = "path"
        choices = [('', f'Please select a opa model {name}')]
        for model in self.opa_model_data:
            if include_id:
                choice_label = f"{model['id']} - {model[field_name]}"
            else:
                choice_label = model[field_name]
            choices.append((model[field_name], choice_label))
        return choices