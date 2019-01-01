BOT_NAME = "dataset"
SPIDER_MODULES = ["dataset.spiders"]
NEWSPIDER_MODULE = "dataset.spiders"
ROBOTSTXT_OBEY = True
FEED_EXPORT_ENCODING = "utf-8"
DEPTH_PRIORITY = 1
SCHEDULER_DISK_QUEUE = "scrapy.squeue.PickleFifoDiskQueue"
SCHEDULER_MEMORY_QUEUE = "scrapy.squeue.FifoMemoryQueue"
