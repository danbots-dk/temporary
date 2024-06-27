from django import template
from django.utils.dateparse import parse_datetime
from django.utils.formats import date_format

register = template.Library()

@register.filter(name='format_api_date')
def format_api_date(value, fmt):
    """Parse a datetime string from the API and format it."""
    dt = parse_datetime(value)
    if dt:
        return date_format(dt, fmt)
    return value