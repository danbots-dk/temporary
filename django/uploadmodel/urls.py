from django.urls import path
from uploadmodel.views import (
    IndexView,
    SimulationModelUploadView,
    WandModelUploadView,
    LogsTemplateView,
    SystemLogDetailView,
    ClearDjangoLogAjaxView,
    ClearTaskLogAjaxView,
    ClearAllLogsAjaxView,
    StitchingSimulationModelUploadView,
    InferenceModelUploadView,
    AjaxLoadModelsView,
    InferenceModelsView,
    SimSegmentStitchingPointCloudView,
    WandXYZDisplacementModelUploadView,
    GetStitchedFileView
)

urlpatterns = [
    path('', IndexView.as_view(), name="index"),
    path('simulation/model/upload', SimulationModelUploadView.as_view(), name="simulation_mode_upload"),
    path('wand/model/upload', WandModelUploadView.as_view(), name="wand_mode_upload"),
    path('logs', LogsTemplateView.as_view(), name="logs"),
    path('log/detail', SystemLogDetailView.as_view(), name="log-detail"),
    path('clear/all/logs', ClearAllLogsAjaxView.as_view(), name="clear-all-logs-ajax"),
    path('clear/django/log', ClearDjangoLogAjaxView.as_view(), name="clear-django-log-ajax"),
    path('clear/task/log', ClearTaskLogAjaxView.as_view(), name="clear-task-log-ajax"),
    path('wand/model/upload', WandModelUploadView.as_view(), name="wand_mode_upload"),
    path('stitching/simulation/model/upload', StitchingSimulationModelUploadView.as_view(), name="stitching_simulation_mode_upload"),
    path('inference-model-upload', InferenceModelUploadView.as_view(), name='inference-model-upload'),
    path('get-models-for-network/<int:network_id>/', AjaxLoadModelsView.as_view(), name='get-models-for-network'),
    path('inference-models', InferenceModelsView.as_view(), name="inference-models"),
    path('simulation/segment/stitching/model/upload', SimSegmentStitchingPointCloudView.as_view(), name="simulation_segment_stitching_model_upload"),
    path('wand/xyz-displacement/model/upload', WandXYZDisplacementModelUploadView.as_view(), name="wand_xyz_displacement_mode_upload"),
    path('get-stitched-files/<int:uploaded_model_id>/', GetStitchedFileView.as_view(), name='get_stitched_files'),
]
