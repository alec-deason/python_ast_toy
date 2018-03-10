import sys
import argparse
import ast
from copy import deepcopy
import math
import random
import inspect

import numpy as np
import imageio

def pixel(x, y, time, width, height):
    """ Calculate the RGB color for the pixel at the given x,y position
    at the given time (ie. frame number in the animation). This will
    draw a 20 pixel diameter black circle centered on (50,50) with a white
    background.

    It's super complicated because I've written it in expectation that it be
    randomly mutated and I need some material for the mutation system to work
    with. Think of all this line noise as genetic diversity or junk DNA in a
    biological system.

    Without the noise, what this code does is calculate the pixel's distance
    from the center of the image and make it black if it's closer than 20 pixels
    and white otherwise. Pre-mutation, it's output does not vary over time.
    """

    c_a = math.sin(1+ time + x + y)
    c_b = math.cos(time + x + y)
    c_d = math.tan(time + x + y)
    x = x - width/2 + time - time + c_a + c_b + c_d - c_a - c_b - c_d
    y = y - height/2 + time - time + c_a + c_b + c_d - c_a - c_b - c_d
    d = math.sqrt(x*x+y*y)

    if d > 20 or d < -20:
        r = 200 + 5 * 10 + 5
        g = 200 + 5 * 10 + 5
        b = 200 + 5 * 10 + 5
    else:
        r = 255 - 200 - 5 * 10 - 5
        g = 255 - 200 - 5 * 10 - 5
        b = 255 - 200 - 5 * 10 - 5

    return r, g, b


class MutateAST(ast.NodeTransformer):
    """ Walk through an AST and wreak havok on it.

    NodeTransformer works by descending through the tree, examining each node
    as it goes and if it has a handler for the node type (see below) it calls
    the handler, which potentially changes the node in some way, otherwise it
    continues to descend.

    I've only implemented handlers for two node types, but there are many more
    that you could operate on. Take a look at
    https://greentreesnakes.readthedocs.io/en/latest/nodes.html for a list of
    node types.
    """

    def visit_Num(self, node):
        """ Handlers are identified by the name of node type they effect, in
        this case ast.Num nodes, or nodes which represent literal numbers in
        the code. A handler can do one of two things, it can return a node in
        which case that node replaces the old one in the AST tree or it can return
        None in which case the node is deleted from the tree.

        Here I change the value of the number by some random amount and return the
        modified node, thus modifying the tree.

        Note that ast.Num nodes are external nodes, they have no children, just
        a value.
        """

        multiplier = 1+((random.random() - 0.5) / 5)
        node.n *= multiplier
        return node

    def visit_BinOp(self, node):
        """ ast.BinOp nodes represent binary operations (like *, / or +). They
        are internal nodes and have two children, `left` and `right`, which
        are the expressions on either side of the operator. I've defined two mutations
        on this node type. One rotates the expressions so that left becomes
        right and right becomes left. The other replaces the operator with a random
        operator.
        """

        # Since we've defined a handler for this node type NodeTransformer won't
        # descend to the node's children on it's own, we need to do that ourselves
        node.right = self.visit(node.right)
        node.left = self.visit(node.left)

        # These mutators are super non-linear and can completely change the
        # nature of the resulting image so I only do them rarely otherwise
        # the output would just be noise (but try playing with the probability
        # of the different mutations to see the effect).
        if random.random() > 0.99:
            node.right, node.left = node.left, node.right

        if random.random() > 0.99:
            new_operator_type = random.choice([
                ast.Sub,
                ast.Add,
                ast.Mult,
                ast.Div,

                # There are a lot of other operators. The ones below tend to
                # produce less interesting effects, at least in the
                # input code I've supplied, but try them out

                #ast.Pow,
                #ast.Mod,
                #ast.LShift,
                #ast.RShift,
                #ast.BitOr,
                #ast.BitXor,
                #ast.BitAnd,
                #ast.FloorDiv,
                ])
            node.op = new_operator_type()
        return node


def calculate_frame(frame_number, pixel_func, frame_size):
    """ Take our random pixel function and use it to calculate a numpy
    3 dimensional array of 8 bit ints which is the format our image
    writer expects.
    """

    image = np.zeros((frame_size[0], frame_size[1], 3))
    for x in range(frame_size[0]):
        for y in range(frame_size[1]):
            # We don't really know if this code will even run, but
            # let's give it a shot. We also don't know what the output
            # will be. Let's make sure that it's an iterable of
            # at least three numeric values because if it isn't
            # there's not much we can do. If the code doesn't work or
            # it's output doesn't meet our base requirements then this
            # will throw an exception which the outer function
            # will handle by rolling back the most recent mutations
            # and trying again.
            color = pixel_func(x, y, frame_number, frame_size[0], frame_size[1])
            color = [float(c) for c in color]
            color = (color[0], color[1], color[2])

            image[x, y, :] = color

    # You need to somehow assure that the output of the randomly mutated
    # code still produces values in the range 0-256 for each color. There
    # are a number of ways of doing that which have different effects.
    # I'll leave a few examples here you can try.
    normalization_technique = "compress"
    if normalization_technique == "compress":
        image = image - image.min()
        image = image / image.max()
        image *= 255
    elif normalization_technique == "mod":
        image = image - image.min()
        image = np.mod(image, 255)
    elif normalization_technique == "clip":
        image = np.max(0, np.min(255, image))

    return image


def do_mutation(ast_tree):
    """ Applies our mutator to the ast_tree and turns the new version.
    Returns the modified tree.
    """

    # NOTE: We copy the tree because our code actually modifies nodes
    # in place.
    # TODO: That's dumb, it should leave the origonal unchanged
    ast_tree = MutateAST().visit(deepcopy(ast_tree))

    return ast_tree

def ast_to_callable(ast_tree):
    """ Convert an ast containing a single function definition into a callable
    version of that function.

    This is the most magical part of this whole program.
    `compile` takes the AST tree and turns it into a code
    object which contains compiled python ready to execute
    but it isn't directly callable. `exec` takes the code object and
    runs it. Since the code object contains a function definition, executing
    it is equivelent to redefining the function. That means that there is now
    a new version of the function defined in this scope.

    Think of this like changing a cell in a jupyter notebook and rerunning it.
    """

    code_object = compile(ast_tree, 'magic_land.py', 'exec')
    exec(code_object)

    # More magic to actually find the new function so we can return it.
    # This is why you normally try not to write code
    # that uses a lot of introspection and metaprogramming.
    function_name = code_object.co_consts[0].co_name
    function = locals()[function_name]

    return function

def main(output_path, seed, size, frame_count):
    print("Using random seed {}".format(seed))

    # Introspect our pixel function to get it's source code, then
    # turn that code into an AST which we'll apply our mutator to
    pixel_function = pixel
    function_source_code = inspect.getsource(pixel_function)
    ast_tree = ast.parse(function_source_code)

    # This is an animation object we can use to write
    # each frame we calculate to disk at a path you specify
    # on the command line. I'd use a path ending in .mp4 This
    # tool can handle other formats but that's the only one I've
    # tested with and basically anything should be able to play
    # mp4 files.
    writer = imageio.get_writer(output_path, fps=24)

    # I like animation loops, so what this does is write out some frames, mutating
    # the pixel function code between each frame and keeping a list of
    # the functions used for each frame. It then works backward over the list
    # of functions writing out additional frames. The effect is that the code
    # get's stranger and stranger over the first half and then slowly returns
    # to it's initial version over the second half ending with the unmodified
    # code. Should always make a pretty nice loop, though some of the mutations
    # are dramatic so it won't be smooth transitions all the way
    # through.
    previous_ast_tree = ast_tree
    pixel_functions = []
    intro_frames = 10
    for frame in range(int((frame_count-intro_frames)/2)):
        success = False
        retries = 10

        print(frame)
        while not success:
            try:
                image = calculate_frame(frame, pixel_function, size)
            except Exception as e:
                print(e)
                # Something about this version of the function doesn't work
                # so roll back to the old version and mutate again in the
                # hopes that it will eventually work.

                if retries > 0:
                    ast_tree = do_mutation(previous_ast_tree)
                    pixel_function = ast_to_callable(ast_tree)
                    retries -= 1
                    continue
                else:
                    # It's possible that the problem is actually caused by a
                    # much earlier mutation in combination with the current
                    # frame number, in which case this simple rollback won't
                    # help. We've tried a few times, now give up.
                    # TODO: deeper rollbacks or some other way have handling this
                    raise

            writer.append_data(image.astype(np.uint8))
            if frame == 0:
                # Repeat the first frame a few times for the sake of
                # a lead in to the loop
                for _ in range(intro_frames):
                    writer.append_data(image.astype(np.uint8))

            # The previous mutation worked (or did after some retries).
            # Make a new mutation and move on to the next frame.
            success = True
            pixel_functions.append(pixel_function)
            prev_ast_tree = ast_tree
            ast_tree = do_mutation(ast_tree)
            pixel_function = ast_to_callable(ast_tree)

    print("Done with the outbound half of the loop, starting on the return")
    last_frame = frame
    while pixel_functions:
        pixel_function = pixel_functions.pop()
        frame += 1
        print(last_frame - (frame - last_frame))
        try:
            image = calculate_frame(frame, pixel_function, size)
        except Exception as e:
            # Just because this function worked without error going forward
            # doesn't mean it will work on the return loop because the frame
            # number has changed. It would be nice to have a cleaner way of
            # handling that than just skiping the frame.
            print(e)
            continue
        writer.append_data(image.astype(np.uint8))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The file to write animation out to. Should end in .mp4")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random number seed")
    parser.add_argument("--size", type=int, nargs=2, default=(112,112), help="Width and height of animation")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames")
    args = parser.parse_args()

    if args.seed is None:
        # If you're reading this Drew, I don't care. I just want seeds that are easy
        # to copy and paste.
        seed = random.randrange(999999)
    else:
        seed = args.seed

    main(args.path, seed, args.size, args.frames)
