import time
from random import randint
import subprocess
import AppOpener
from pynput.keyboard import Key, Controller

class SystemTasks:
    def __init__(self):
        self.keyboard = Controller()

    def write(self, text):
        self.keyboard.type(text)

    def select(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('a')
        self.keyboard.release('a')
        self.keyboard.release(Key.ctrl)

    def hitEnter(self):
        self.keyboard.press(Key.enter)
        self.keyboard.release(Key.enter)

    def delete(self):
        self.select()
        self.keyboard.press(Key.backspace)
        self.keyboard.release(Key.backspace)

    def copy(self):
        self.select()
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('c')
        self.keyboard.release('c')
        self.keyboard.release(Key.ctrl)

    def paste(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('v')
        self.keyboard.release('v')
        self.keyboard.release(Key.ctrl)

    def new_file(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('n')
        self.keyboard.release('n')
        self.keyboard.release(Key.ctrl)

    def save(self, name):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('s')
        self.keyboard.release('s')
        self.keyboard.release(Key.ctrl)
        time.sleep(0.2)
        self.write(name)
        self.hitEnter()


class TabOpt:
    def __init__(self):
        self.keyboard = Controller()

    def switchTab(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press(Key.tab)
        self.keyboard.release(Key.tab)
        self.keyboard.release(Key.ctrl)

    def closeTab(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('w')
        self.keyboard.release('w')
        self.keyboard.release(Key.ctrl)

    def newTab(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('t')
        self.keyboard.release('t')
        self.keyboard.release(Key.ctrl)


class WindowOpt:
    def __init__(self):
        self.keyboard = Controller()

    def closeWindow(self):
        self.keyboard.press(Key.alt_l)
        self.keyboard.press(Key.f4)
        self.keyboard.release(Key.f4)
        self.keyboard.release(Key.alt_l)

    def minimizeWindow(self):
        for i in range(2):
            self.keyboard.press(Key.cmd)
            self.keyboard.press(Key.down)
            self.keyboard.release(Key.down)
            self.keyboard.release(Key.cmd)
            time.sleep(0.05)

    def maximizeWindow(self):
        self.keyboard.press(Key.cmd)
        self.keyboard.press(Key.up)
        self.keyboard.release(Key.up)
        self.keyboard.release(Key.cmd)

    def switchWindow(self):
        self.keyboard.press(Key.alt_l)
        self.keyboard.press(Key.tab)
        self.keyboard.release(Key.tab)
        self.keyboard.release(Key.alt_l)

def app_path(app):
    app_paths = {'access': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\ACCICONS.exe',
                 'power point': 'C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs',
                 'word': 'C:\\ProgramData\Microsoft\\Windows\\Start Menu\\Programs',
                 'excel': 'C:\\ProgramData\Microsoft\\Windows\\Start Menu\\Programs',
                 'outlook': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\OUTLOOK.exe',
                 'onenote': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\ONENOTE.exe',
                 'publisher': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\MSPUB.exe',
                 'sharepoint': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\GROOVE.exe',
                 'infopath designer': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\INFOPATH.exe',
                 'infopath filler': 'C:\\Program Files (x86)\\Microsoft Office\\Office14\\INFOPATH.exe'
                }
    try:
        return app_paths[app]
    except KeyError:
        return None


def open_app(query):
    ms_office = ('access', 'power point','visual studio', 'word', 'excel', 'outlook', 'onenote', 'publisher', 'sharepoint', 'infopath designer','infopath filler','whatsapp')
    for app in ms_office:
        if app in query:
            path = app_path(app)
            subprocess.Popen(path)
            return True
    AppOpener.run(query[5:])
    return True

def take_note(note):
    open_app("open notepad")
    time.sleep(0.2)
    sys_task = SystemTasks()
    sys_task.write(note)
    sys_task.save(f'note_{randint(1, 100)}')