import pyadin
import queue

if __name__ == "__main__":
    q = queue.Queue()
    pipeline = pyadin.setup_pipeline(q)
    pipeline.open()
    while pipeline.update() is not None:
        while q.empty() is False:
            audioseg = q.get()
            if audioseg['is_end'] is False:
                # some processes here
                pass
            del audioseg
    pipeline.close()
