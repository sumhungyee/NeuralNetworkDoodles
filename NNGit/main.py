
from NN import *
from pickle_initialiser import *
from utils import *



network = Network(ROWS * COLS, 10, 4)
pictures = getfile()

gridlist = list(map(lambda pic: np.array(convert(pic.grid)).flatten(), pictures))
expected = list(map(lambda pic: EXPECTED[CONVERSION[pic.classification]], pictures))
network.mass_train(gridlist, expected, LEARNRATE)

######
######
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DRAWER")


def init_grid(rows, cols, colour):
    return [[colour for i in range(cols)] for j in range(rows)]

def draw(win, grid):
    win.fill(BG_COLOUR)
    draw_grid(win, grid)

    pygame.display.update()

def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(win, pixel, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
    if DRAW_GRID_LINES:
        for i in range(ROWS + 1):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE), (WIDTH, i * PIXEL_SIZE))
        for i in range(COLS + 1):
            pygame.draw.line(win, BLACK, (i * PIXEL_SIZE, 0), (i * PIXEL_SIZE, HEIGHT - TOOLBAR_HEIGHT))

def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE
    if row >= ROWS:
        raise IndexError
    
    return row, col

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BG_COLOUR)
drawing_colour = DARK
exit_via_1 = False

while run:
    
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    if pygame.mouse.get_pressed()[0]:
        try: 
            row, col = get_row_col_from_pos(pygame.mouse.get_pos())
            grid[row][col] = drawing_colour
            if row - 1 >= 0:
                grid[row-1][col] = tuple(map(lambda x: sum(x)/2, list(zip(grid[row-1][col], drawing_colour))))
            if row + 1 < ROWS:
                grid[row+1][col] = tuple(map(lambda x: sum(x)/2, list(zip(grid[row+1][col], drawing_colour))))
            if col - 1 >= 0:
                grid[row][col-1] = tuple(map(lambda x: sum(x)/2, list(zip(grid[row][col-1], drawing_colour))))
            if col + 1 < COLS:
                grid[row][col+1] = tuple(map(lambda x: sum(x)/2, list(zip(grid[row-1][col+1], drawing_colour))))

        except IndexError:
            pass
    if pygame.mouse.get_pressed()[1]:
        exit_via_1 = True
        run = False
        
        
    if pygame.mouse.get_pressed()[2]:
        try: 
            row, col = get_row_col_from_pos(pygame.mouse.get_pos())
            grid[row][col] = WHITE
            if row - 1 >= 0:
                grid[row-1][col] = WHITE
            if row + 1 < ROWS:
                grid[row+1][col] = WHITE
            if col - 1 >= 0:
                grid[row][col-1] = WHITE
            if col + 1 < COLS:
                grid[row][col+1] = WHITE

        except IndexError:
            pass

    draw(WIN, grid)

pygame.quit()

if exit_via_1:
    data = convert(grid)
    data = np.array(data).flatten()
    feed = list(network.forward_prop(data))

    print(f"Oh! This is a {SHAPES[feed.index(max(feed))]}!")
    correct = input("Was I right? (Y/N) ")

    store = True
    error = True
    while error:
        if correct == "Y":
            print("Of course I'm right.")

            correctindex = feed.index(max(feed))
            

            error = False
        elif correct == "N":
            error = False

            possibly_correct_index = input(f""":( What is it then? 
            1: {SHAPES[0]}
            2: {SHAPES[1]}
            3: {SHAPES[2]}
            4: {SHAPES[3]}
            esc: you made a mistake
            """)
            if possibly_correct_index.isdigit() and 1 <= int(possibly_correct_index) <= 4:
                correctindex = int(possibly_correct_index) - 1
                if correctindex == feed.index(max(feed)):
                    print("But that means I was right...?")

            
            elif possibly_correct_index == "esc":
                print("Alright bye.")
                store = False
                break
            else:
                print("Come again?")
                correct = input("Was I right? (Y/N) ")
                continue


        else:
            print("Sorry, I didn't quite catch that.")
            correct = input("Was I right? (Y/N) ")
            continue
        
        if store:
            file = open("data.txt", "rb")
            data = pickle.load(file)
            data.append(Picture(grid, f"{SHAPES[correctindex]}"))
            print(f"Stored in repository as a {SHAPES[correctindex]}")
            file.close()
            dump = open("data.txt", "wb")
            pickle.dump(data, dump)
            dump.close()
                

  
  
   
   
  


