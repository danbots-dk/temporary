import os
import logging
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.urls import reverse_lazy
from django.views.generic.edit import FormView
from django.views import View
from django.views.generic import TemplateView
from utils.constant import (
    NETWORK_MODEL_BY_NETWORK_API, 
    NETWORK_MODEL_API, 
    NETWORK_API, 
    INFERENCE_MODEL_API,
    LUM_MODEL,
    DEPTH_MODEL,
    WRAP_MODEL
    )
from conf.settings import NNDB_SERVER, INFERENCE_SERVER
from uploadmodel.forms import (
    UploadSimulationModelForm, 
    UploadWandModelForm, 
    UploadStitchingSimulationModelForm, 
    InferUploadedModelForm,
    SimSegmentStitchingPointCloudForm,
    UploadWandXYZDisplacementModelForm
)
from django.contrib import messages
from uploadmodel.tasks import (
    simulation_model_task, 
    wand_model_task, 
    stitching_simulation_model_task, 
    inference_upload_model_task,
    segment_stitching_simulation_model_task,
    wand_model_task_3d,
    updated_stitching_simulation_model_task
)
from .wand import Wand
from .wand_printer import WandPrinter
from django.conf import settings
from pathlib import Path
import requests
from django.shortcuts import render, redirect
import moonrakerpy as moonpy
import time
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexView(LoginRequiredMixin, TemplateView):
    template_name = 'uploadmodel/index.html'


class SimulationModelUploadView(LoginRequiredMixin, FormView):
    form_class = UploadSimulationModelForm
    template_name = 'uploadmodel/simulation-create.html' 
    success_url = reverse_lazy('simulation_mode_upload') 

    def form_valid(self, form):
        # Process the form data here
        input_folder = form.cleaned_data['input_folder']
        batch_name = form.cleaned_data['batch_name']
        uploaded_model = form.cleaned_data['uploaded_model']
        depth_model = form.cleaned_data['depth_model']
        wrap_model = form.cleaned_data['wrap_model']
        lum_model = form.cleaned_data['lum_model']
        opa_model = form.cleaned_data['opa_model']
        start = form.cleaned_data['start']
        stop = form.cleaned_data['stop']
        step = form.cleaned_data['step']
        # print(input_folder, batch_name, uploaded_model, depth_model, wrap_model, start, stop, step)
        success_message = "The simulation model has been processed successfully."
        messages.success(self.request, success_message)
        if all([uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step]):
            simulation_model_task(
                uploaded_model=uploaded_model,
                wrap_model=wrap_model,
                depth_model=depth_model,
                lum_model=lum_model,
                opa_model=opa_model,
                batch_name=batch_name,
                input_folder=input_folder,
                start=start,
                stop=stop,
                step=step
            )
        return super().form_valid(form)


class WandModelUploadView(LoginRequiredMixin, FormView):
    form_class = UploadWandModelForm
    template_name = 'uploadmodel/wand-create.html' 
    success_url = reverse_lazy('wand_mode_upload') 

    def form_valid(self, form):
        # Process the form data here
        wand_ip = form.cleaned_data['wand_ip']
        dataset_size = form.cleaned_data['dataset_size']
        input_folder = form.cleaned_data['input_folder']
        batch_name = form.cleaned_data['batch_name']
        uploaded_model = form.cleaned_data['uploaded_model']
        depth_model = form.cleaned_data['depth_model']
        wrap_model = form.cleaned_data['wrap_model']
        lum_model = form.cleaned_data['lum_model']
        opa_model = form.cleaned_data['opa_model']
        start = int(0)
        stop = int(dataset_size)
        step = int(1)
        dias = form.cleaned_data['dias']
        flash = form.cleaned_data['flash']
        depth_comparison = form.cleaned_data['depth_comparison']
        
        if wand_ip and dataset_size and dias is not None and flash is not None:
            wand_obj = Wand(
                wand_ip=wand_ip, 
                dataset_size=int(dataset_size), 
                base_path=input_folder,
                flash=flash,
                dias=dias
                )
            folder = wand_obj.collect_dataset()
            print(folder)
            if os.path.exists(folder):
                success_message = "The wand device model has been processed successfully."
                messages.success(self.request, success_message)
                
                if all(x is not None for x in [uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder]) and start >= 0 and stop > start and step > 0:
                    wand_model_task(
                        uploaded_model=uploaded_model,
                        wrap_model=wrap_model,
                        depth_model=depth_model,
                        lum_model=lum_model,
                        opa_model=opa_model,
                        batch_name=batch_name,
                        input_folder=folder,
                        start=start,
                        stop=stop,
                        step=step,
                        depth_comparison=depth_comparison
                    )
            else:        
                error_message = "Something went wrong while taking pictrues or folder does not exist"
                messages.error(self.request, error_message)
        else:
            # print(input_folder, batch_name, uploaded_model, depth_model, wrap_model, start, stop, step)
            success_message = "The wand model has been processed successfully."
            messages.success(self.request, success_message)
            if all(x is not None for x in [uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder]) and start >= 0 and stop > start and step > 0:
                wand_model_task(
                    uploaded_model=uploaded_model,
                    wrap_model=wrap_model,
                    depth_model=depth_model,
                    lum_model=lum_model,
                    opa_model=opa_model,
                    batch_name=batch_name,
                    input_folder=input_folder,
                    start=start,
                    stop=stop,
                    step=step,
                    depth_comparison=depth_comparison
                )
        return super().form_valid(form)
    


class LogsTemplateView(LoginRequiredMixin, TemplateView):
    template_name = "uploadmodel/logs.html"


class SystemLogDetailView(TemplateView):
    """
    Upload Model Log Detail View
    """
    template_name = 'uploadmodel/system-log-details.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        type = self.request.GET.get('type')
        if type == "upload_model":
            log_file_path = settings.BASE_DIR.joinpath('task_stdout.log')
        elif type == "django":
            log_file_path = settings.BASE_DIR.joinpath('django_stdout.log')
        else:
            log_file_path = settings.BASE_DIR.joinpath('task_output.log')
        try:
            with open(log_file_path, 'r') as log_file:
                log_lines = log_file.readlines()
            log_lines.reverse()
            log_content = ''.join(log_lines)
        except FileNotFoundError:
            log_content = "Log file not found."
        context['log_content'] = log_content
        context['type'] = type
        return context

class ClearDjangoLogAjaxView(View):

    def get(self, request, *args, **kwargs):
        if self.request.user.is_authenticated and self.request.headers.get('x-requested-with') == 'XMLHttpRequest':
            log_file_path = settings.BASE_DIR.joinpath('django_stdout.log')
            try:
                with open(log_file_path, 'w') as log_file:
                    log_file.write('')
                log_content = "Log file content has been cleared."
                response_data = {"result": "success", "message": log_content}
                return JsonResponse(response_data, status=200, safe=False)
            except FileNotFoundError:
                response_data = {"result": "failed", "message": "Log file not found"}
                return JsonResponse(response_data, status=200)
        else:
            response_data = {"result": "failed", "message": "Invalid request."}
            return JsonResponse(response_data, status=200)

class ClearTaskLogAjaxView(View):

    def get(self, request, *args, **kwargs):
        if self.request.user.is_authenticated and self.request.headers.get('x-requested-with') == 'XMLHttpRequest':
            log_file_path = settings.BASE_DIR.joinpath('task_stdout.log')
            try:
                with open(log_file_path, 'w') as log_file:
                    log_file.write('')
                log_content = "Log file content has been cleared."
                response_data = {"result": "success", "message": log_content}
                return JsonResponse(response_data, status=200, safe=False)
            except FileNotFoundError:
                response_data = {"result": "failed", "message": "Log file not found"}
                return JsonResponse(response_data, status=200)
        else:
            response_data = {"result": "failed", "message": "Invalid request."}
            return JsonResponse(response_data, status=200)


class ClearAllLogsAjaxView(View):

    def get(self, request, *args, **kwargs):
        if self.request.user.is_authenticated and self.request.headers.get('x-requested-with') == 'XMLHttpRequest':
            task_file_path = settings.BASE_DIR.joinpath('task_stdout.log')
            django_file_path = settings.BASE_DIR.joinpath('django_stdout.log')
            task_output_file_path = settings.BASE_DIR.joinpath('task_output.log')
            log = settings.BASE_DIR.joinpath('log.log')
            try:
                with open(task_file_path, 'w') as log_file:
                    log_file.write('')
                with open(django_file_path, 'w') as log_file:
                    log_file.write('')
                with open(task_output_file_path, 'w') as log_file:
                    log_file.write('')
                with open(log, 'w') as log_file:
                    log_file.write('')

                log_content = "Log file content has been cleared."
                response_data = {"result": "success", "message": log_content}
                return JsonResponse(response_data, status=200, safe=False)
            except FileNotFoundError:
                response_data = {"result": "failed", "message": "Log file not found"}
                return JsonResponse(response_data, status=200)
        else:
            response_data = {"result": "failed", "message": "Invalid request."}
            return JsonResponse(response_data, status=200)


class StitchingSimulationModelUploadView(LoginRequiredMixin, FormView):
    form_class = UploadStitchingSimulationModelForm
    template_name = 'uploadmodel/stitching-simulation-create.html' 
    success_url = reverse_lazy('stitching_simulation_mode_upload') 

    def form_valid(self, form):
        # Process the form data here
        input_folder = form.cleaned_data['input_folder']
        batch_name = form.cleaned_data['batch_name']
        uploaded_model = form.cleaned_data['uploaded_model']
        depth_model = form.cleaned_data['depth_model']
        wrap_model = form.cleaned_data['wrap_model']
        lum_model = form.cleaned_data['lum_model']
        opa_model = form.cleaned_data['opa_model']
        start = int(0)
        start = form.cleaned_data['start']
        stop = form.cleaned_data['stop']
        step = form.cleaned_data['step']

        nb_points = form.cleaned_data['nb_points']
        radius = form.cleaned_data['radius']
        voxel_size = form.cleaned_data['voxel_size']


        # print(input_folder, batch_name, uploaded_model, depth_model, wrap_model, start, stop, step)
        success_message = "The simulation model has been processed successfully."
        messages.success(self.request, success_message)
        if all([uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder, start, stop, step]):
            stitching_simulation_model_task(
                uploaded_model=uploaded_model,
                wrap_model=wrap_model,
                depth_model=depth_model,
                lum_model=lum_model,
                opa_model=opa_model,
                batch_name=batch_name,
                input_folder=input_folder,
                start=start,
                stop=stop,
                step=step,
                nb_points=int(nb_points),
                radius=float(radius),
                voxel_size=float(voxel_size)
            )
           
        return super().form_valid(form)

class InferenceModelUploadView(View):
    template_name = 'uploadmodel/inference-model-create.html'

    def get(self, request, *args, **kwargs):
        form = InferUploadedModelForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = InferUploadedModelForm(request.POST, request=request)
        if not form.is_valid():
            return render(request, self.template_name, {'form': form})

        net_value = form.cleaned_data['net']
        model_value = form.cleaned_data['model']
        is_currently_used = True

        try:
            network_data = self.fetch_api_data(f"{NNDB_SERVER}{NETWORK_API}{net_value}/")
            models_data = self.fetch_api_data(f"{NNDB_SERVER}{NETWORK_MODEL_API}{model_value}/")
            model_file_path = Path(models_data['model_path']).with_suffix('.h5')
            nndb_url_api = f"{NNDB_SERVER}{INFERENCE_MODEL_API}"

            model_type_map = {
                'depth': (DEPTH_MODEL, 'depth_model'),
                'wrap': (WRAP_MODEL, 'wrap_model'),
                'lum': (LUM_MODEL, 'lum_model'),
            }

            model_type_key = network_data.get('type')
            if model_type_key in model_type_map:
                model_type, file_key = model_type_map[model_type_key]
                inference_upload_model_task(
                    infer_url=INFERENCE_SERVER + model_type,
                    nndb_url=nndb_url_api,
                    model_path=str(model_file_path),
                    file_key=file_key,
                    net=net_value, 
                    model=model_value,
                    is_currently_used=is_currently_used
                )  
                messages.success(request, "The model has been successfully processed and uploaded for inference.")
            else:
                messages.error(request, "Unsupported network type.")
        except Exception as e:
            messages.error(request, f"An error occurred: {e}")

        return redirect('inference-model-upload')

    def fetch_api_data(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

class AjaxLoadModelsView(View):
    def get(self, request, network_id):
        api_url = f"{NNDB_SERVER}{NETWORK_MODEL_BY_NETWORK_API}{network_id}/"
        try:
            # Send a GET request to the external API
            response = requests.get(api_url)
            response.raise_for_status()  # This will raise an error for 4xx/5xx responses

            models_data = response.json()
            models_data_formatted = [(model['id'], model['model_path']) for model in models_data if 'id' in model and 'model_path' in model]
            return JsonResponse(models_data_formatted, safe=False)
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch models from external API: {e}")
            return JsonResponse({'error': 'Failed to fetch models from the external API'}, status=400)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return JsonResponse({'error': 'An unexpected error occurred'}, status=500)


class InferenceModelsView(LoginRequiredMixin, TemplateView):
    template_name = "uploadmodel/inference-models.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        api_url = f"{NNDB_SERVER}{INFERENCE_MODEL_API}"
        response = requests.get(api_url)
        response.raise_for_status() 
        inferenc_models_data = response.json()
        context['inference_models'] = inferenc_models_data
        return context


class SimSegmentStitchingPointCloudView(View):
    template_name = 'uploadmodel/stitching-segment-simulation-create.html'

    def get(self, request, *args, **kwargs):
        form = SimSegmentStitchingPointCloudForm(request=request)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = SimSegmentStitchingPointCloudForm(request.POST, request=request)
        if not form.is_valid():
            return render(request, self.template_name, {'form': form})

        model = form.cleaned_data['uploaded_model']
        stitched_files = form.cleaned_data['stitched_file']
        batch_name = form.cleaned_data.get('batch_name')
        nb_points = form.cleaned_data.get('nb_points')
        radius = form.cleaned_data.get('radius')
        voxel_size = form.cleaned_data.get('voxel_size')

        # Handle multiple stitched files and their associated names
        missing_ply_files = os.path.join(settings.BASE_DIR, 'missing_ply_files_for_segmentstit.log') 
        stitched_file_details = []
        for file in stitched_files:
            file_name = request.POST.get(f'stitched_file_name_{file}')
            output_folder_with_file = os.path.join(file, 'external.ply')
            if os.path.isfile(output_folder_with_file):
                stitched_file_details.append({'output_folder': output_folder_with_file, 'name': file_name})
            else:
                with open(missing_ply_files, 'a') as log_file:
                    log_file.write(f"Missing file: {output_folder_with_file} for the name: {file_name}\n")
        if len(stitched_file_details) > 0:
            success_message = "The simulation segment stitching has been processed successfully."
            messages.success(self.request, success_message)
            segment_stitching_simulation_model_task(
                model=model,
                batch_name=batch_name,
                point_cloud_files=stitched_file_details,
                nb_points=int(nb_points),
                radius=float(radius),
                voxel_size=float(voxel_size)
            ) 
        else:
            success_message = "Segment element or name is empty."
            messages.error(self.request, success_message)
            print("Segmented point cloud file is emply")
        return redirect('simulation_segment_stitching_model_upload')
    
class WandXYZDisplacementModelUploadView(LoginRequiredMixin, FormView):
    form_class = UploadWandXYZDisplacementModelForm
    template_name = 'uploadmodel/wand-xyz-displacement-create.html' 
    success_url = reverse_lazy('wand_xyz_displacement_mode_upload') 

    def form_valid(self, form):
        # Process the form data here
        wand_ip = form.cleaned_data['wand_ip']
        dataset_size = form.cleaned_data['dataset_size']
        input_folder = form.cleaned_data['input_folder']
        batch_name = form.cleaned_data['batch_name']
        uploaded_model = form.cleaned_data['uploaded_model']
        depth_model = form.cleaned_data['depth_model']
        wrap_model = form.cleaned_data['wrap_model']
        lum_model = form.cleaned_data['lum_model']
        opa_model = form.cleaned_data['opa_model']
        start = int(0)
        stop = int(dataset_size)
        step = int(1)
        dias = form.cleaned_data['dias']
        flash = form.cleaned_data['flash']
        depth_comparison = form.cleaned_data['depth_comparison']
        x_displacement = form.cleaned_data['x_displacement']
        y_displacement = form.cleaned_data['y_displacement']
        z_displacement = form.cleaned_data['z_displacement']

        
        if wand_ip and dataset_size and dias is not None and flash is not None:

            wand_obj = WandPrinter(
                wand_ip=wand_ip, 
                dataset_size=int(dataset_size), 
                base_path=input_folder,
                flash=flash,
                dias=dias,
                x_displacement=x_displacement,
                y_displacement=y_displacement,
                z_displacement=z_displacement
                )
            folder = wand_obj.collect_dataset()
            print(folder)
            if os.path.exists(folder):
                success_message = "The wand device model has been processed successfully."
                messages.success(self.request, success_message)
                
                if all(x is not None for x in [uploaded_model, wrap_model, depth_model, lum_model, opa_model, batch_name, input_folder]) and start >= 0 and stop > start and step > 0:
                    wand_model_task_3d(
                        upload_model_id=uploaded_model,
                        wrap_model=wrap_model,
                        depth_model=depth_model,
                        lum_model=lum_model,
                        opa_model=opa_model,
                        batch_name=batch_name,
                        input_folder=folder,
                        start=int(start),
                        stop=int(stop),
                        step=int(step),
                        depth_comparison=depth_comparison,
                        x_displacement=str(x_displacement),
                        y_displacement=str(y_displacement),
                        z_displacement=str(z_displacement)
                    )
            else:        
                error_message = "Something went wrong while taking pictrues or folder does not exist"
                messages.error(self.request, error_message)
        else:
            # print(input_folder, batch_name, uploaded_model, depth_model, wrap_model, start, stop, step)
            success_message = "The wand model has been processed successfully."
            messages.success(self.request, success_message)
        return super().form_valid(form)

class GetStitchedFileView(View):
    def get(self, request, uploaded_model_id, *args, **kwargs):
        api_url = f"{NNDB_SERVER}apis/v1/model/{uploaded_model_id}/batch-name-stitching-simulation-point-cloud"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            stitched_files = response.json()
            stitched_file_choices = [{'output_folder': cat['output_folder'], 'name': cat['name']} for cat in stitched_files]
            return JsonResponse({'stitched_file_choices': stitched_file_choices})
        except requests.exceptions.RequestException as e:
            logger.exception("Error fetching stitched files: %s", e)
            return JsonResponse({'error': 'Error fetching stitched files'}, status=500)
