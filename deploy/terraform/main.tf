# PostgreSQL for the services (RDS, Multi-AZ, encrypted, IAM auth, credentials in Secrets
# Manager) and the image registry. The EKS cluster and VPC are shared platform resources.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.80" }
  }
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags { tags = { service = "campaign-budget-optimizer", environment = var.environment } }
}

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "vpc_id" {
  type = string
}

variable "database_subnet_ids" {
  type = list(string)
}

variable "eks_node_security_group_id" {
  type = string
}

locals {
  name = "cbo-${var.environment}"
}

resource "aws_ecr_repository" "app" {
  name                 = "campaign-budget-optimizer"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_kms_key" "db" {
  description         = "${local.name} database"
  enable_key_rotation = true
}

resource "aws_db_subnet_group" "db" {
  name       = local.name
  subnet_ids = var.database_subnet_ids
}

resource "aws_security_group" "db" {
  name_prefix = "${local.name}-db-"
  vpc_id      = var.vpc_id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.eks_node_security_group_id]
  }
}

resource "aws_db_parameter_group" "db" {
  name   = local.name
  family = "postgres16"
  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }
  parameter {
    name  = "log_min_duration_statement"
    value = "250" # log queries slow enough to threaten the 300 ms latency objective
  }
}

resource "aws_db_instance" "db" {
  identifier                          = local.name
  engine                              = "postgres"
  engine_version                      = "16"
  instance_class                      = "db.r6g.large"
  allocated_storage                   = 50
  max_allocated_storage               = 300
  db_name                             = "cbo"
  username                            = "cbo"
  manage_master_user_password         = true
  storage_encrypted                   = true
  kms_key_id                          = aws_kms_key.db.arn
  multi_az                            = true
  db_subnet_group_name                = aws_db_subnet_group.db.name
  vpc_security_group_ids              = [aws_security_group.db.id]
  parameter_group_name                = aws_db_parameter_group.db.name
  backup_retention_period             = 14
  deletion_protection                 = true
  skip_final_snapshot                 = false
  final_snapshot_identifier           = "${local.name}-final"
  iam_database_authentication_enabled = true
  performance_insights_enabled        = true
  performance_insights_kms_key_id     = aws_kms_key.db.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "database_endpoint" {
  value = aws_db_instance.db.endpoint
}
