### 运行tensorboard
在终端执行
```
tensorboard --logdir=runs --port=6006 --bind_all
```
然后访问
```
192.168.7.232:6006
```

rsync -avz --progress --remove-source-files /path/to/source/ /path/to/destination/