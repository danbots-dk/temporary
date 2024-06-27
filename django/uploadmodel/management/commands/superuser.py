import os
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from conf.settings import SUPERUSER_USERNAME, SUPERUSER_PASS

class Command(BaseCommand):
    help = 'Creating superuser'

    def handle(self, *args, **kwargs):
      
        user_data = User.objects.filter(username=SUPERUSER_USERNAME)
        if user_data.exists():
            self.stdout.write("Username already exists in the system")
            return
        else:
            try:
                created_user = User.objects.create_superuser(
                    username=SUPERUSER_USERNAME,
                    password=SUPERUSER_PASS,
                    is_staff=True,
                    is_active=True,
                    is_superuser=True
                )
                self.stdout.write(
                    self.style.SUCCESS('Superuser successfully created with username: %s' % created_user.username)
                )
            except Exception as e:
                self.stdout.write(self.style.ERROR('Error: %s' % (e,)))
     