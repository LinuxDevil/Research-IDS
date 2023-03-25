import os
import smbprotocol


def send_file(smbclient, file_path, dest_path):
    with open(file_path, 'rb') as f:
        smbclient.create_file(dest_path)
        smbclient.write(dest_path, f.read())
        smbclient.close(dest_path)


def send_directory(smbclient, dir_path, dest_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir_path)
            dest_file_path = os.path.join(dest_path, rel_path)
            send_file(smbclient, file_path, dest_file_path)


def receive_file(smbclient, file_path, dest_path):
    with open(dest_path, 'wb') as f:
        data = smbclient.read(file_path)
        f.write(data)


def receive_directory(smbclient, dir_path, dest_path):
    for file_info in smbclient.list_path(dir_path):
        if file_info.is_directory():
            if file_info.filename not in ('.', '..'):
                sub_dir_path = os.path.join(dir_path, file_info.filename)
                sub_dest_path = os.path.join(dest_path, file_info.filename)
                os.makedirs(sub_dest_path, exist_ok=True)
                receive_directory(smbclient, sub_dir_path, sub_dest_path)
        else:
            file_path = os.path.join(dir_path, file_info.filename)
            dest_file_path = os.path.join(dest_path, file_info.filename)
            receive_file(smbclient, file_path, dest_file_path)


def transfer_files(source_path, dest_path, host, username, password):
    smbclient = smbprotocol.SMB(username, password, host, conn_port=445)

    if os.path.isdir(source_path):
        smbclient.create_directory(dest_path)
        send_directory(smbclient, source_path, dest_path)
    elif os.path.isfile(source_path):
        send_file(smbclient, source_path, dest_path)
    else:
        raise ValueError("Invalid path: {}".format(source_path))

    smbclient.close()

    if os.path.isdir(source_path):
        smbclient = smbprotocol.SMB(username, password, host, conn_port=445)
        receive_directory(smbclient, dest_path, source_path)
        smbclient.close()
    elif os.path.isfile(source_path):
        smbclient = smbprotocol.SMB(username, password, host, conn_port=445)
        receive_file(smbclient, dest_path, source_path)
        smbclient.close()


# Example usage:
transfer_files('/path/to/source', '\\\\remote_host\\share\\path\\to\\destination', 'remote_host', 'username',
               'password')
