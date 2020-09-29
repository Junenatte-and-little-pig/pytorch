# -*- encoding: utf-8 -*-
import os
import pkgutil
import re

import requests
from torchvision import models


def main():
    for _, name, ispkg in pkgutil.iter_modules(models.__path__):
        if not ispkg and 'utils' not in name:
            data = pkgutil.get_data(models.__name__, '{}.py'.format(name))
            # print(data.decode())
            data_str = data.decode()
            compiler = re.compile(r'https://[\S]*.pth')
            match = re.findall(compiler, data_str)
            for link in match:
                filepath = r'E:\Github\pytorch\models'
                filename = str.rsplit(link, '/')[-1]
                # print(filename)
                try:
                    f = requests.get(link)
                    path=os.path.join(filepath, filename)
                    if not os.path.exists(path):
                        with open(path, 'wb') as model:
                            model.write(f.content)
                        print(filename + " done.")
                        f.close()
                    else:
                        print(filename+' has downloaded.')
                except requests.exceptions.RequestException as e:
                    print(e)
                    continue
            # print(match)


if __name__ == '__main__':
    main()
