# Run Django migrations
python manage.py makemigrations --merge --noinput
python manage.py migrate
python manage.py superuser

# Collect static files (if necessary)
python manage.py collectstatic --noinput