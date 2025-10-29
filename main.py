from acl_anthology import Anthology

def main():
    print("hello world")

    anthology = Anthology.from_repo()
    event = anthology.get_event("aacl-2023")
    for volume in event.volumes():
        print(volume.title)
if __name__ == "__main__":
    main()
