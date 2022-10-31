import os


def file_check(project: str):
    project_net_config = os.path.join('project', project, 'net_config.yaml')
    project_augment_config_yaml = os.path.join('project', project, 'augment_config.yaml')

    if not os.path.exists(project_net_config):
        ...
    if not os.path.exists(project_augment_config_yaml):
        ...


if __name__ == '__main__':
    project = 'demo'
    file_check(project)
