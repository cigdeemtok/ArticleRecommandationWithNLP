# Generated by Django 4.1.2 on 2024-05-22 01:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("articleapp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Dataset",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                ("title", models.CharField(max_length=255)),
                ("abstract", models.TextField()),
                ("fulltext", models.TextField()),
                ("keywords", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="Makale",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                ("title", models.CharField(max_length=255)),
                ("abstract", models.TextField()),
                ("fulltext", models.TextField()),
                ("keywords", models.TextField()),
            ],
        ),
    ]